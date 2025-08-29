# STDA-SwiFT (Official Repository)

This is the official repository for the paper:

Whole-brain Transferable Representations from Large-Scale fMRI Data Improve Task-Evoked Brain Activity Decoding ([arXiv:2507.22378](https://arxiv.org/abs/2507.22378)).

It provides the STDA-SwiFT pretrained encoder and a minimal inference example for generating embeddings from 4D fMRI volumes.

## Install

```bash
pip install -r requirements.txt
# optional: install package locally for imports
pip install -e .
```

## Usage

Quick start (auto-detect seq_len from input shape):

```bash
python examples/infer.py --input /path/to/volume.npy --ckpt ckpt/pretrained.ckpt
```

- `ckpt/pretrained.ckpt` is the pretrained task-fMRI encoder.
- The script prints the embedding vector shape and a preview of values.
- Input may be `.npy` (H,W,D,T) or `.nii/.nii.gz` with 4D data.
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

Time embeddings will be interpolated when `--seq_len` differs from training.

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
