import argparse
import os
import torch

# Local import fallback so examples run without package install
try:
    from stdaswift.models import SwinTransformer4D
    from stdaswift.utils.io import load_input
except Exception:
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from stdaswift.models import SwinTransformer4D
    from stdaswift.utils.io import load_input


def detect_seq_len(input_path: str) -> int:
    if input_path.endswith('.npy'):
        import numpy as np
        arr = np.load(input_path, mmap_mode='r')
        assert arr.ndim == 4, f"Expected 4D array (H,W,D,T), got {arr.shape}"
        return int(arr.shape[3])
    else:
        try:
            import nibabel as nib
        except Exception:
            raise RuntimeError('nibabel not installed; install it or provide a .npy input')
        img = nib.load(input_path)
        shape = img.shape
        assert len(shape) == 4, f"Expected 4D NIfTI (H,W,D,T), got {shape}"
        return int(shape[3])


def build_encoder(seq_len: int):
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


def load_encoder_weights(encoder: torch.nn.Module, ckpt_path: str):
    ckpt = torch.load(ckpt_path, map_location='cpu')
    state = ckpt.get('state_dict', ckpt)
    # auto-handle common prefixes transparently
    new_state = {}
    for k, v in state.items():
        if k.startswith('encoder.'):
            new_state[k.replace('encoder.', '')] = v
        elif k.startswith('model.'):
            new_state[k.replace('model.', '')] = v
        else:
            new_state[k] = v
    encoder.load_state_dict(new_state, strict=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help='Path to .npy or NIfTI 4D (H,W,D,T)')
    parser.add_argument('--ckpt', default=os.path.join(os.path.dirname(__file__), '..', 'ckpt', 'pretrained.ckpt'),
                        help='Path to encoder state_dict or Lightning ckpt. Defaults to release ckpt.')
    parser.add_argument('--seq_len', type=int, default=None, help='If omitted, inferred from input T')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    seq_len = args.seq_len or detect_seq_len(args.input)
    device = torch.device(args.device)

    encoder = build_encoder(seq_len).to(device)
    if args.ckpt and os.path.exists(args.ckpt):
        load_encoder_weights(encoder, args.ckpt)
        if hasattr(encoder, 'interpolate_time_embed'):
            encoder.interpolate_time_embed(seq_len)

    x = load_input(args.input, (96, 96, 96, seq_len)).to(device)
    encoder.eval()
    with torch.no_grad():
        feat = encoder(x)
        b, c, h, w, d, t = feat.shape
        feat = feat.view(b, c, h * w * d * t).mean(dim=2)
    print('Feature shape:', feat.shape)
    print(feat[0, :5].cpu().numpy())


if __name__ == '__main__':
    main()


