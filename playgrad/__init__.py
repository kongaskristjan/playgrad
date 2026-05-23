"""Playgrad — visualization library for deep learning experiments.

The library provides a `Session` that hooks into a PyTorch model and
publishes per-batch activation/gradient snapshots for inspection in a
web UI. It deliberately contains no training logic; training lives in
`examples/`.
"""

from __future__ import annotations

from playgrad.schedule import BatchPosition, Schedule
from playgrad.session import BatchSnapshot, Mode, Session, start

__all__ = [
    "BatchPosition",
    "BatchSnapshot",
    "Mode",
    "Schedule",
    "Session",
    "start",
]
