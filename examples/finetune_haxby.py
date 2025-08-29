import argparse
import os
from typing import List, Tuple, Dict

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

# Local import fallback so examples run without package install
try:
    from stdaswift.models import SwinTransformer4D
except Exception:
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from stdaswift.models import SwinTransformer4D


def build_encoder(seq_len: int) -> SwinTransformer4D:
    model = SwinTransformer4D(
        img_size=(96, 96, 96, seq_len),
        in_chans=1,
        embed_dim=36,
        window_size=(6, 6, 6, 1),
        first_window_size=(6, 6, 6, 1),
        patch_size=(6, 6, 6, 1),
        depths=(2, 2, 18, 2),
        num_heads=(3, 6, 12, 24),
        downsample="mergingv2",
    )
    return model


def load_encoder_weights(encoder: nn.Module, ckpt_path: str):
    if ckpt_path is None or ckpt_path == "" or not os.path.exists(ckpt_path):
        return
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt.get("state_dict", ckpt)
    new_state = {}
    for k, v in state.items():
        if k.startswith("encoder."):
            new_state[k.replace("encoder.", "")] = v
        elif k.startswith("model."):
            new_state[k.replace("model.", "")] = v
        else:
            new_state[k] = v
    encoder.load_state_dict(new_state, strict=False)


def _read_haxby_labels(targets_txt_path: str) -> List[str]:
    # Load CSV-like file with numpy to avoid requiring pandas
    # The file has headers, including a column named "labels" in most releases
    rec = np.recfromcsv(targets_txt_path)
    if "labels" in rec.dtype.names:
        labels = rec["labels"].astype(str).tolist()
    elif "stim" in rec.dtype.names:
        labels = rec["stim"].astype(str).tolist()
    else:
        # fallback: use the last column
        labels = rec.tolist()
        if isinstance(labels[0], tuple):
            labels = [str(x[-1]) for x in labels]
        else:
            labels = [str(x) for x in labels]
    return labels


def _get_label_mapping(labels: List[str], ignore_label: str = "rest") -> Dict[str, int]:
    unique = sorted({l for l in labels if l != ignore_label})
    return {lab: idx for idx, lab in enumerate(unique)}


def _resize_spatial(arr: np.ndarray, target_spatial: Tuple[int, int, int]) -> np.ndarray:
    # arr: (H, W, D, T) float32
    assert arr.ndim == 4
    h, w, d, t = arr.shape
    target_h, target_w, target_d = target_spatial
    out = np.empty((target_h, target_w, target_d, t), dtype=np.float32)
    for i in range(t):
        vol_hwd = torch.from_numpy(arr[..., i]).float()  # (H, W, D)
        vol_dhw = vol_hwd.permute(2, 0, 1).unsqueeze(0).unsqueeze(0)  # (1,1,D,H,W)
        vol_resized = torch.nn.functional.interpolate(
            vol_dhw,
            size=(target_d, target_h, target_w),
            mode="trilinear",
            align_corners=False,
        )
        vol_hwd_resized = vol_resized.squeeze(0).squeeze(0).permute(1, 2, 0)  # (H', W', D')
        out[..., i] = vol_hwd_resized.numpy()
    return out


class HaxbyVolumeDataset(Dataset):
    def __init__(
        self,
        subject: int = 1,
        seq_len: int = 1,
        categories: List[str] = None,
        split: str = "train",
        val_fraction: float = 0.2,
        random_state: int = 0,
    ) -> None:
        super().__init__()
        try:
            from nilearn import datasets, image
        except Exception as e:
            raise RuntimeError(
                "nilearn is required for this example. Please `pip install nilearn`."
            ) from e

        rng = np.random.RandomState(random_state)
        data = datasets.fetch_haxby(subjects=[subject])
        func_path = data.func[0]
        targets_path = data.session_target[0] if hasattr(data, "session_target") else data.session_target

        labels_all = _read_haxby_labels(targets_path)

        # Load image as float32 array (H,W,D,T)
        img = image.load_img(func_path)
        arr = img.get_fdata().astype(np.float32)
        # Basic standardization (per-volume z-score across voxels)
        arr = (arr - np.mean(arr, axis=(0, 1, 2), keepdims=True)) / (
            np.std(arr, axis=(0, 1, 2), keepdims=True) + 1e-6
        )

        # Resize spatial dims to (96,96,96)
        arr = _resize_spatial(arr, (96, 96, 96))

        # Filter labels and build mapping
        if categories is None:
            categories = [
                "face",
                "cat",
                "house",
                "chair",
                "bottle",
                "scissors",
                "shoe",
                "scrambledpix",
            ]
        label_to_idx = {lab: i for i, lab in enumerate(categories)}

        # Build samples as contiguous windows of length seq_len with constant label (non-rest)
        X: List[Tuple[int, int]] = []  # list of (start_idx, label_idx)
        t_total = arr.shape[3]
        for start in range(0, t_total - seq_len + 1):
            window_labels = labels_all[start : start + seq_len]
            if any(lab not in label_to_idx for lab in window_labels):
                continue
            if not all(l == window_labels[0] for l in window_labels):
                continue
            X.append((start, label_to_idx[window_labels[0]]))

        # Train/val split by index
        indices = np.arange(len(X))
        rng.shuffle(indices)
        split_at = int((1.0 - val_fraction) * len(indices))
        if split == "train":
            indices = indices[:split_at]
        else:
            indices = indices[split_at:]

        self.arr = arr
        self.seq_len = seq_len
        self.samples = [X[i] for i in indices]
        self.num_classes = len(categories)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        start, y = self.samples[idx]
        h, w, d, _ = self.arr.shape
        t = self.seq_len
        vol = self.arr[:, :, :, start : start + t]  # (H,W,D,T)
        vol = torch.from_numpy(vol).float().unsqueeze(0).unsqueeze(0)  # (1,1,H,W,D,T)
        return vol, torch.tensor(y, dtype=torch.long)


class EncoderClassifier(nn.Module):
    def __init__(self, encoder: SwinTransformer4D, num_classes: int) -> None:
        super().__init__()
        self.encoder = encoder
        self.pool = nn.AdaptiveAvgPool1d(1)
        feature_dim = encoder.num_features
        self.classifier = nn.Linear(feature_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,1,H,W,D,T)
        feat = self.encoder(x)  # (B, C, D', H', W', T')
        b, c, d, h, w, t = feat.shape
        feat = feat.view(b, c, d * h * w * t)
        feat = self.pool(feat).squeeze(-1)
        logits = self.classifier(feat)
        return logits


def train_one_epoch(model, loader, device, optimizer, criterion):
    model.train()
    total = 0
    correct = 0
    loss_sum = 0.0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        loss_sum += float(loss.item()) * x.size(0)
        pred = logits.argmax(dim=1)
        correct += int((pred == y).sum().item())
        total += int(x.size(0))
    return loss_sum / max(total, 1), correct / max(total, 1)


@torch.no_grad()
def eval_one_epoch(model, loader, device, criterion):
    model.eval()
    total = 0
    correct = 0
    loss_sum = 0.0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss_sum += float(loss.item()) * x.size(0)
        pred = logits.argmax(dim=1)
        correct += int((pred == y).sum().item())
        total += int(x.size(0))
    return loss_sum / max(total, 1), correct / max(total, 1)


def main():
    parser = argparse.ArgumentParser(description="Finetune STDA-SwiFT on Haxby classification")
    parser.add_argument("--ckpt", default=os.path.join(os.path.dirname(__file__), "..", "ckpt", "pretrained.ckpt"))
    parser.add_argument("--subject", type=int, default=1)
    parser.add_argument("--seq_len", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--freeze_encoder", action="store_true")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num_workers", type=int, default=2)
    args = parser.parse_args()

    device = torch.device(args.device)

    # Data
    train_ds = HaxbyVolumeDataset(subject=args.subject, seq_len=args.seq_len, split="train")
    val_ds = HaxbyVolumeDataset(subject=args.subject, seq_len=args.seq_len, split="val")

    print(f"Train dataset size: {len(train_ds)}")
    print(f"Val dataset size: {len(val_ds)}")
    
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True
    )

    # Model
    encoder = build_encoder(seq_len=args.seq_len)
    load_encoder_weights(encoder, args.ckpt)
    if hasattr(encoder, "interpolate_time_embed"):
        encoder.interpolate_time_embed(args.seq_len)

    model = EncoderClassifier(encoder, num_classes=train_ds.num_classes).to(device)
    if args.freeze_encoder:
        for p in model.encoder.parameters():
            p.requires_grad = False

    # Optimizer/scheduler
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, device, optimizer, criterion)
        val_loss, val_acc = eval_one_epoch(model, val_loader, device, criterion)
        scheduler.step()
        print(
            f"Epoch {epoch:03d} | train_loss={train_loss:.4f} acc={train_acc:.3f} | val_loss={val_loss:.4f} acc={val_acc:.3f}"
        )
        if val_acc > best_val_acc:
            best_val_acc = val_acc

    print(f"Best val acc: {best_val_acc:.3f}")


if __name__ == "__main__":
    main()


