import argparse
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True, help="Path to Lightning checkpoint")
    parser.add_argument("--out", required=True, help="Output .pt path for encoder state dict")
    args = parser.parse_args()

    ckpt = torch.load(args.ckpt, map_location="cpu")
    state = ckpt.get("state_dict", ckpt)
    enc = {}
    for k, v in state.items():
        if k.startswith("encoder."):
            enc[k.replace("encoder.", "")] = v
        elif k.startswith("model."):
            enc[k.replace("model.", "")] = v
    torch.save(enc, args.out)
    print(f"Saved encoder state_dict to {args.out} with {len(enc)} tensors")


if __name__ == "__main__":
    main()


