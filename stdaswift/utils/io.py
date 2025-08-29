import torch
import numpy as np
from typing import Tuple

try:
    import nibabel as nib
except Exception:
    nib = None


def load_input(path: str, target_shape: Tuple[int, int, int, int]) -> torch.Tensor:
    """
    Load an input volume sequence as torch tensor of shape (1, 1, H, W, D, T).
    Supports .npy (H, W, D, T) and .nii/.nii.gz (H, W, D, T) when 4D.
    """
    h, w, d, t = target_shape
    if path.endswith(".npy"):
        arr = np.load(path)
    else:
        assert nib is not None, "Please install nibabel to read NIfTI files."
        img = nib.load(path)
        arr = img.get_fdata()

    assert arr.ndim == 4, f"Expected 4D data (H, W, D, T), got {arr.shape}"
    assert arr.shape == (h, w, d, t), f"Expected {(h,w,d,t)}, got {arr.shape}"
    arr = arr.astype(np.float32)
    tensor = torch.from_numpy(arr)[None, None]  # (1,1,H,W,D,T)
    return tensor


