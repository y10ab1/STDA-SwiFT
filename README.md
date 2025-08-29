# STDA-SwiFT (Official Repository)

This is the official repository for the paper:

Whole-brain Transferable Representations from Large-Scale fMRI Data Improve Task-Evoked Brain Activity Decoding ([arXiv:2507.22378](https://arxiv.org/abs/2507.22378)).

It provides the STDA-SwiFT pretrained encoder and a minimal inference example for generating embeddings from 4D fMRI volumes.

## Install

```bash
pip install -r requirements.txt
# Note: You do NOT need to install the package to run the examples.
```

## Quick start

```bash
python examples/infer.py --input /path/to/volume.npy --ckpt ckpt/pretrained.ckpt
```

## Finetune on Haxby classification (example)

```bash
pip install nilearn
python examples/finetune_haxby.py --ckpt ckpt/pretrained.ckpt \
  --subject 1 --seq_len 1 --batch_size 20 --epochs 10 --lr 1e-4
```

- Finetunes the encoder on Haxby categories
  using contiguous windows of length `--seq_len` with constant labels and resized to `(96,96,96)`.
- Requires internet to download Haxby via `nilearn` on first run.
- We highly recommend standardizing the fMRI data to `zscore_sample` before training, you can do this by running `clean_img(img, standardize='zscore_sample')` as in the `examples/finetune_haxby.py` script.
- You can modify `--seq_len` to leverage temporal context; the script will interpolate the time
  embeddings to the requested length.

Both example scripts include a local import fallback so they work out-of-the-box from the repo
without running `pip install -e .`. If you prefer package-style imports in your own project,
you can still `pip install -e .` after cloning.

- `ckpt/pretrained.ckpt` is the pretrained task-fMRI encoder.
- The script prints the embedding vector shape and a preview of values.
- Input may be `.npy`, `.pt`, or `.nii/.nii.gz` with 4D (H,W,D,T) fMRI data.
 - `--seq_len` is optional; if omitted, the script infers it from input.
- We recommend padding the input to `(96,96,96,T)` before any further training or inference.

## Checkpoint format

The loader accepts either:
- A plain `state_dict` of the encoder, or
- A Lightning checkpoint with keys prefixed by `encoder.` or `model.`

## Model

`stdaswift.models.SwinTransformer4D` with:
- `img_size=(96,96,96,T)`
- `patch_size=(6,6,6,1)`
- `depths=(2,2,18,2)`
- `num_heads=(3,6,12,24)`
- `window_size=(6,6,6,1)`

## Citation

If you find this work useful, please cite:

Peng, Y.-P., Cheung, V.K.M., Su, L. Whole-brain Transferable Representations from Large-Scale fMRI Data Improve Task-Evoked Brain Activity Decoding. arXiv:2507.22378, 2025. ([link](https://arxiv.org/abs/2507.22378))

BibTeX:

```
@article{Peng2025STDA-SwiFT,
  title={Whole-brain Transferable Representations from Large-Scale fMRI Data Improve Task-Evoked Brain Activity Decoding},
  author={Yueh-Po Peng and Vincent K. M. Cheung and Li Su},
  journal={arXiv preprint arXiv:2507.22378},
  year={2025}
}
```
