# playgrad

A playground for hand-rolled deep learning models in PyTorch.

The first project is a small CIFAR-style residual convolutional network trained
on CIFAR10.

## Setup

```bash
uv sync
```

## Train CIFAR10

```bash
uv run python scripts/train_cifar10.py --epochs 50
```

Useful flags:

- `--batch-size` (default `256`).
- `--blocks-per-stage` — depth knob; total depth is `6n + 2` (default `3` gives ResNet-20).
- `--lr`, `--momentum`, `--weight-decay` — SGD hyperparameters.
- `--device` — `cpu`, `cuda`, or `mps`. Auto-detected when omitted.
- `--bf16` — wrap forward/loss in `torch.autocast` with `bfloat16` (no `GradScaler` needed).
- `--checkpoint path/to/file.pt` — save best-by-test-accuracy weights.

The script uses SGD with Nesterov momentum, cosine LR annealing, and the
standard CIFAR10 augmentations (random crop with 4-pixel padding + horizontal
flip + normalisation).

## Architecture

`lib/resnet.py` defines a CIFAR ResNet:

- 3x3 stem conv into 16 channels.
- Three stages of `BasicBlock`s at widths `(16, 32, 64)` with `stride=2`
  downsampling between stages. The shortcut uses a 1x1 conv when the shapes
  change, otherwise identity.
- Global average pool into a linear classifier.

`resnet20()` is a convenience constructor (`blocks_per_stage=3`,
~270k parameters).

## Tests

```bash
uv run pytest
uv run ty check
```
