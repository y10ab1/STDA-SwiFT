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
python examples/finetune_haxby.py --ckpt ckpt/pretrained.ckpt \
  --subject 1 --seq_len 1 --batch_size 20 --epochs 10 --lr 1e-4
```

- The script finetunes the encoder on Haxby categories.
- Requires internet to download Haxby via `nilearn` on first run.
- We highly recommend standardizing the fMRI data to `zscore_sample` before training, you can do this by running `clean_img(img, standardize='zscore_sample')` as in the `examples/finetune_haxby.py` script.



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
