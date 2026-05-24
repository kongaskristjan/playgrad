"""Render snapshot tensors to PNG bytes for the UI.

The library captures full per-batch tensors; the renderer takes a per-sample
slice and produces one horizontal strip per layer for the right pane of the
UI. Conv-style activations become a row of square channel tiles; 1D
activations become a single short heatmap row.

Sequential (grayscale) colormap for activations; diverging (blue-white-red)
for gradients. PNGs are encoded with `compress_level=1` to favour speed over
size — bytes travel a local WebSocket, so wire size is irrelevant.
"""

from __future__ import annotations

import io
from typing import Literal

import numpy as np
from PIL import Image
from torch import Tensor
from torch.nn import functional as F

TILE_SIZE: int = 128
TILE_GAP: int = 2
LINEAR_TILE_HEIGHT: int = 32
LINEAR_MAX_BINS: int = 256
LINEAR_BIN_WIDTH: int = 16
PNG_COMPRESS_LEVEL: int = 1

ColormapKind = Literal["activation", "gradient"]


def render_strip(
    tensor: Tensor | None,
    sample_idx: int,
    *,
    kind: ColormapKind,
) -> bytes | None:
    """Render a per-channel horizontal strip as PNG bytes.

    Returns `None` if the tensor is `None`, `sample_idx` is out of range, or
    the per-sample shape is unsupported (anything other than `[C, H, W]` or
    `[F]`).
    """
    if tensor is None or tensor.ndim == 0:
        return None
    if not 0 <= sample_idx < tensor.shape[0]:
        return None
    sample = tensor[sample_idx]
    if sample.ndim == 3:
        return _render_chw(sample, kind=kind)
    if sample.ndim == 1:
        return _render_1d(sample, kind=kind)
    return None


def _render_chw(tensor: Tensor, *, kind: ColormapKind) -> bytes:
    c, h, w = tensor.shape
    mode = "nearest" if max(h, w) <= TILE_SIZE else "area"
    resized = F.interpolate(
        tensor.unsqueeze(0).float(),
        size=(TILE_SIZE, TILE_SIZE),
        mode=mode,
    )[0]
    rgb = _apply_colormap(resized.numpy(), kind=kind)
    strip = _concat_tiles_with_gaps(list(rgb), TILE_GAP)
    return _encode_png(strip)


def _concat_tiles_with_gaps(tiles: list[np.ndarray], gap: int) -> np.ndarray:
    if gap <= 0 or len(tiles) <= 1:
        return np.concatenate(tiles, axis=1)
    h = tiles[0].shape[0]
    spacer = np.full((h, gap, 3), 255, dtype=np.uint8)
    pieces: list[np.ndarray] = []
    for i, tile in enumerate(tiles):
        if i > 0:
            pieces.append(spacer)
        pieces.append(tile)
    return np.concatenate(pieces, axis=1)


def _render_1d(tensor: Tensor, *, kind: ColormapKind) -> bytes:
    values = tensor.float()
    f = values.shape[0]
    if f > LINEAR_MAX_BINS:
        values = F.adaptive_avg_pool1d(
            values.view(1, 1, f), LINEAR_MAX_BINS
        ).view(-1)
        f = LINEAR_MAX_BINS
    rgb_row = _apply_colormap(values.numpy(), kind=kind)
    image = np.broadcast_to(rgb_row[None, :, :], (LINEAR_TILE_HEIGHT, f, 3)).copy()
    pil = Image.fromarray(image, mode="RGB").resize(
        (f * LINEAR_BIN_WIDTH, LINEAR_TILE_HEIGHT), Image.Resampling.NEAREST
    )
    return _pil_to_png(pil)


def _apply_colormap(values: np.ndarray, *, kind: ColormapKind) -> np.ndarray:
    if kind == "activation":
        return _sequential_colormap(values)
    return _diverging_colormap(values)


def _sequential_colormap(values: np.ndarray) -> np.ndarray:
    lo, hi = float(values.min()), float(values.max())
    if hi <= lo:
        gray = np.zeros(values.shape, dtype=np.uint8)
    else:
        gray = (((values - lo) / (hi - lo)) * 255).astype(np.uint8)
    return np.stack([gray, gray, gray], axis=-1)


def _diverging_colormap(values: np.ndarray) -> np.ndarray:
    abs_max = float(max(abs(values.min()), abs(values.max()), 1e-12))
    norm = (values / abs_max).clip(-1.0, 1.0)
    rgb = np.full(norm.shape + (3,), 255, dtype=np.uint8)
    pos = norm > 0
    neg = norm < 0
    fade_pos = (255 * (1 - norm[pos])).astype(np.uint8)
    rgb[pos, 1] = fade_pos
    rgb[pos, 2] = fade_pos
    fade_neg = (255 * (1 + norm[neg])).astype(np.uint8)
    rgb[neg, 0] = fade_neg
    rgb[neg, 1] = fade_neg
    return rgb


def _encode_png(rgb: np.ndarray) -> bytes:
    return _pil_to_png(Image.fromarray(rgb, mode="RGB"))


def _pil_to_png(pil: Image.Image) -> bytes:
    buf = io.BytesIO()
    pil.save(buf, format="PNG", compress_level=PNG_COMPRESS_LEVEL)
    return buf.getvalue()
