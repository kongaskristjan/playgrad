"""Tests for the Mermaid graph builder."""

from __future__ import annotations

from torch import nn

from playgrad.ui.graph import CONFIG_HEADER, build_mermaid


class TwoLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
        )
        self.fc = nn.Linear(8, 4)

    def forward(self, x):  # pragma: no cover - structural test only
        return self.fc(self.stem(x).mean(dim=(-1, -2)))


def test_includes_config_header_and_flowchart() -> None:
    src = build_mermaid(TwoLayer())
    assert src.startswith(CONFIG_HEADER)
    assert "flowchart TD" in src


def test_has_node_per_submodule() -> None:
    src = build_mermaid(TwoLayer())
    for name in ("stem", "stem.0", "stem.1", "stem.2", "fc"):
        assert name in src


def test_has_parent_to_child_edges() -> None:
    src = build_mermaid(TwoLayer())
    # nn.Sequential children → stem.0/1/2
    assert "stem --> stem_0" in src
    assert "stem --> stem_1" in src
    assert "stem --> stem_2" in src
    # root → top-level children
    assert "root --> stem" in src
    assert "root --> fc" in src


def test_node_labels_include_class_name() -> None:
    src = build_mermaid(TwoLayer())
    assert "BatchNorm2d" in src
    assert "Conv2d" in src
    assert "Linear" in src
