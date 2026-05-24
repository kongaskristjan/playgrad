# playgrad

A visualization library for deep learning experiments (work in progress) and a
playground for hand-rolled PyTorch models.

## Layout

- `playgrad/` — the visualization library. Intended to be `pip`-installable;
  contains no training logic. Currently a stub.
- `examples/` — runnable Python examples, each in its own subdirectory and
  fully containing its training logic.
- `tests/` — tests for both the `playgrad` library and the examples; the
  layout mirrors the source tree.

## Setup

```bash
uv sync
```

## CIFAR10 example

The first example is a small CIFAR-style residual convolutional network
trained on CIFAR10 (`examples/cifar10/`).

```bash
uv run python -m examples.cifar10.main --epochs 50
```

Useful flags:

- `--batch-size` (default `256`).
- `--blocks-per-stage` — depth knob; total depth is `6n + 2` (default `3` gives ResNet-20).
- `--lr`, `--momentum`, `--weight-decay` — SGD hyperparameters.
- `--device` — `cpu`, `cuda`, or `mps`. Auto-detected when omitted.
- `--bf16` — wrap forward/loss in `torch.autocast` with `bfloat16` (no `GradScaler` needed).
- `--checkpoint path/to/file.pt` — save best-by-test-accuracy weights.
- `--playgrad-port 8080` — launch the playgrad UI on this port. Training
  pauses on the first batch; open the URL to drive it with the step / detach
  controls.

The script uses SGD with Nesterov momentum, cosine LR annealing, and the
standard CIFAR10 augmentations (random crop with 4-pixel padding + horizontal
flip + normalisation).

### Architecture

`examples/cifar10/resnet.py` defines a pre-activation CIFAR ResNet (ResNet v2,
He et al. 2016) with ResNet-D-style downsampling shortcuts (He et al. 2018):

- 3x3 stem conv into 16 channels.
- Three stages of `PreActBlock`s at widths `(16, 32, 64)`. Each block is
  `BN -> ReLU -> 3x3 conv -> BN -> ReLU -> 3x3 conv`, then adds the shortcut,
  keeping the identity path free of nonlinearities and batch-norms.
- Stage transitions downsample with `stride=2` and use the ResNet-D shortcut:
  a 2x2 average pool followed by a 1x1 conv (no info-losing strided 1x1, no
  extra BN on the shortcut path).
- A final BN + ReLU before global average pool, then a linear classifier.

`resnet20()` is a convenience constructor (`blocks_per_stage=3`,
~270k parameters).

## Using the `playgrad` library

```python
import playgrad

session = playgrad.start(model, epochs=50, phases={"train": 196, "val": 40})
playgrad.serve(
    session,
    port=8080,
    # Optional: denormalize input images for display (e.g., CIFAR10 stats).
    input_mean=(0.4914, 0.4822, 0.4465),
    input_std=(0.2470, 0.2435, 0.2616),
)

for epoch in range(50):
    for batch in train_loader:
        with session.batch(phase="train", epoch=epoch):
            optimizer.zero_grad()
            loss = ...
            loss.backward()
            optimizer.step()
    for batch in val_loader:
        with session.batch(phase="val", epoch=epoch):
            ...

session.close()  # UI keeps running so you can browse the last snapshot
```

Open `http://localhost:8080` while training is running. The top bar drives
the session with five "go" buttons — `stop`, `step batch`, `step epoch`,
`step until end`, `step until custom` (opens a dialog where you pick the
target phase / epoch / batch) — and `detach` (run unattended without
further pauses). The leading icon button toggles the architecture pane;
a trailing icon button toggles the input-image pane. The left pane shows
the module hierarchy as a Mermaid diagram; clicking a node scrolls the
matching layer card into view, and clicking a card's header scrolls the
diagram to its node — both ends briefly flash so it's clear what got
hit. The centre pane shows one card per
submodule with horizontally-scrollable activation and activation-gradient
strips for the selected sample; the right pane shows the input image for
that sample (RGB or grayscale), denormalized with the `input_mean` /
`input_std` passed to `serve()` if any.

See `INTERNALS.md` for the architecture overview.

## Tests

```bash
uv run pytest
uv run ty check
```
