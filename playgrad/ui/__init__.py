"""Playgrad UI: NiceGUI app + tensor-to-PNG rendering + Mermaid graph."""

from __future__ import annotations

from playgrad.ui.app import serve
from playgrad.ui.graph import build_mermaid
from playgrad.ui.render import render_strip

__all__ = ["build_mermaid", "render_strip", "serve"]
