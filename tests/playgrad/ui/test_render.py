"""Tests for tensor → PNG rendering."""

from __future__ import annotations

import io

import pytest
import torch
from PIL import Image

from playgrad.ui.render import (
    LINEAR_BIN_WIDTH,
    LINEAR_MAX_BINS,
    LINEAR_TILE_HEIGHT,
    TILE_SIZE,
    ColormapKind,
    render_strip,
)


def _decode(png: bytes) -> Image.Image:
    return Image.open(io.BytesIO(png)).convert("RGB")


@pytest.mark.parametrize("kind", ["activation", "gradient"])
def test_chw_strip_dimensions(kind: ColormapKind) -> None:
    tensor = torch.randn(4, 8, 32, 32)
    png = render_strip(tensor, sample_idx=2, kind=kind)
    assert png is not None
    img = _decode(png)
    assert img.size == (8 * TILE_SIZE, TILE_SIZE)


@pytest.mark.parametrize("kind", ["activation", "gradient"])
def test_1d_strip_dimensions(kind: ColormapKind) -> None:
    tensor = torch.randn(4, 10)
    png = render_strip(tensor, sample_idx=0, kind=kind)
    assert png is not None
    img = _decode(png)
    assert img.size == (10 * LINEAR_BIN_WIDTH, LINEAR_TILE_HEIGHT)


def test_1d_strip_caps_at_max_bins() -> None:
    tensor = torch.randn(4, LINEAR_MAX_BINS * 4)
    png = render_strip(tensor, sample_idx=0, kind="activation")
    assert png is not None
    img = _decode(png)
    assert img.size == (LINEAR_MAX_BINS * LINEAR_BIN_WIDTH, LINEAR_TILE_HEIGHT)


def test_returns_none_for_none_tensor() -> None:
    assert render_strip(None, sample_idx=0, kind="activation") is None


def test_returns_none_for_out_of_range_sample() -> None:
    tensor = torch.randn(4, 8, 32, 32)
    assert render_strip(tensor, sample_idx=10, kind="activation") is None
    assert render_strip(tensor, sample_idx=-1, kind="activation") is None


def test_returns_none_for_unsupported_shape() -> None:
    # Per-sample shape would be [3, 4, 5, 6] — 4D, not supported.
    tensor = torch.randn(2, 3, 4, 5, 6)
    assert render_strip(tensor, sample_idx=0, kind="activation") is None


def test_zero_variance_tensor_renders() -> None:
    tensor = torch.zeros(2, 4, 8, 8)
    png = render_strip(tensor, sample_idx=0, kind="activation")
    assert png is not None
    img = _decode(png)
    assert img.size == (4 * TILE_SIZE, TILE_SIZE)


def test_gradient_zero_center_renders() -> None:
    # All-zero gradient must not crash the diverging colormap (abs_max=0 edge).
    tensor = torch.zeros(2, 4, 8, 8)
    png = render_strip(tensor, sample_idx=0, kind="gradient")
    assert png is not None
