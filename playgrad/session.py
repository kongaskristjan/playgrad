"""Playgrad session: state machine, hook installation, snapshot publishing.

A `Session` is created once per training run via `playgrad.start(...)`.
The user wraps each batch with `with session.batch(phase=..., epoch=...)`:

    session = playgrad.start(model, epochs=50, phases={"train": 196, "val": 40})
    for epoch in range(50):
        for batch in train_loader:
            with session.batch(phase="train", epoch=epoch):
                optimizer.zero_grad()
                loss = ...
                loss.backward()
                optimizer.step()

Because `optimizer.zero_grad()` lives at the start of the user's batch body,
parameter `.grad` is still populated when the context manager exits, so the
session reads it straight off the model — no backward hooks needed.

The UI (added later) drives the session by calling `stop`, `step_batch`,
`step_phase`, `step_epoch`, `continue_run`, and finally `close`. Whether the
session captures activations/gradients for a given batch is decided up-front
at `__enter__` from the schedule + current mode, so forward hooks are only
installed for batches that will actually be inspected.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from enum import StrEnum
from types import TracebackType
from typing import Self

from torch import Tensor, nn
from torch.utils.hooks import RemovableHandle

from playgrad.schedule import BatchPosition, Schedule


class Mode(StrEnum):
    STEP = "step"
    UNTIL_PHASE_CHANGE = "until_phase_change"
    UNTIL_EPOCH_CHANGE = "until_epoch_change"
    RUN = "run"


@dataclass(frozen=True)
class BatchSnapshot:
    position: BatchPosition
    activations: dict[str, Tensor]
    gradients: dict[str, Tensor]


class Session:
    def __init__(
        self,
        model: nn.Module,
        *,
        epochs: int,
        phases: dict[str, int],
    ) -> None:
        self.model = model
        self._schedule = Schedule(epochs=epochs, phases=phases)
        self._mode: Mode = Mode.STEP
        self._cv = threading.Condition()
        self._resume_token = 0
        self._pause_count = 0
        self._closed = False
        self._activations: dict[str, Tensor] = {}
        self._hook_handles: list[RemovableHandle] = []
        self._snapshot: BatchSnapshot | None = None

    @property
    def schedule(self) -> Schedule:
        return self._schedule

    @property
    def mode(self) -> Mode:
        with self._cv:
            return self._mode

    @property
    def snapshot(self) -> BatchSnapshot | None:
        return self._snapshot

    @property
    def closed(self) -> bool:
        with self._cv:
            return self._closed

    @property
    def pause_count(self) -> int:
        with self._cv:
            return self._pause_count

    def batch(self, *, phase: str, epoch: int) -> _BatchContext:
        return _BatchContext(self, phase=phase, epoch=epoch)

    def set_schedule(
        self,
        *,
        epochs: int | None = None,
        phases: dict[str, int] | None = None,
    ) -> None:
        with self._cv:
            self._schedule.update(epochs=epochs, phases=phases)

    def stop(self) -> None:
        self._set_mode(Mode.STEP, resume=False)

    def step_batch(self) -> None:
        self._set_mode(Mode.STEP, resume=True)

    def step_phase(self) -> None:
        self._set_mode(Mode.UNTIL_PHASE_CHANGE, resume=True)

    def step_epoch(self) -> None:
        self._set_mode(Mode.UNTIL_EPOCH_CHANGE, resume=True)

    def continue_run(self) -> None:
        self._set_mode(Mode.RUN, resume=True)

    def close(self) -> None:
        with self._cv:
            self._closed = True
            self._cv.notify_all()

    def wait_until_paused(
        self,
        *,
        after_pauses: int = 0,
        timeout: float | None = None,
    ) -> bool:
        with self._cv:
            return self._cv.wait_for(
                lambda: self._pause_count > after_pauses or self._closed,
                timeout=timeout,
            )

    def _set_mode(self, mode: Mode, *, resume: bool) -> None:
        with self._cv:
            self._mode = mode
            if resume:
                self._resume_token += 1
                self._cv.notify_all()

    def _should_capture(self, pos: BatchPosition) -> bool:
        with self._cv:
            if self._closed:
                return False
            mode = self._mode
        match mode:
            case Mode.STEP:
                return True
            case Mode.UNTIL_PHASE_CHANGE:
                return pos.is_last_in_phase
            case Mode.UNTIL_EPOCH_CHANGE:
                return pos.is_last_in_epoch
            case Mode.RUN:
                return False

    def _install_hooks(self) -> None:
        self._activations.clear()
        for name, module in self.model.named_modules():
            if module is self.model:
                continue
            handle = module.register_forward_hook(self._make_hook(name))
            self._hook_handles.append(handle)

    def _remove_hooks(self) -> None:
        for h in self._hook_handles:
            h.remove()
        self._hook_handles.clear()

    def _make_hook(self, name: str):
        def hook(_module: nn.Module, _inputs: object, output: object) -> None:
            if isinstance(output, Tensor):
                self._activations[name] = output

        return hook

    def _publish_snapshot(self, pos: BatchPosition) -> None:
        gradients = {
            name: param.grad
            for name, param in self.model.named_parameters()
            if param.grad is not None
        }
        self._snapshot = BatchSnapshot(
            position=pos,
            activations=dict(self._activations),
            gradients=gradients,
        )

    def _wait_for_proceed(self) -> None:
        with self._cv:
            seen = self._resume_token
            self._pause_count += 1
            self._cv.notify_all()
            while self._resume_token == seen and not self._closed:
                self._cv.wait()


class _BatchContext:
    def __init__(self, session: Session, *, phase: str, epoch: int) -> None:
        self._session = session
        self._phase = phase
        self._epoch = epoch
        self._position: BatchPosition | None = None
        self._captured = False

    def __enter__(self) -> Self:
        if self._session.closed:
            return self
        self._position = self._session._schedule.advance(self._phase, self._epoch)
        self._captured = self._session._should_capture(self._position)
        if self._captured:
            self._session._install_hooks()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        if self._position is None or not self._captured:
            return
        self._session._remove_hooks()
        if exc is None and not self._session.closed:
            self._session._publish_snapshot(self._position)
            self._session._wait_for_proceed()
        self._session._activations.clear()

    @property
    def position(self) -> BatchPosition | None:
        return self._position

    @property
    def captured(self) -> bool:
        return self._captured


def start(
    model: nn.Module,
    *,
    epochs: int,
    phases: dict[str, int],
) -> Session:
    return Session(model, epochs=epochs, phases=phases)
