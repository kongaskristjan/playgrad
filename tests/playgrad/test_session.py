"""Tests for the Session state machine and per-batch capture."""

from __future__ import annotations

import threading

import pytest
import torch
from torch import Tensor, nn

import playgrad
from playgrad.session import Mode, Session


class TinyNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(4, 8)
        self.fc2 = nn.Linear(8, 3)

    def forward(self, x: Tensor) -> Tensor:
        return self.fc2(torch.relu(self.fc1(x)))


def _train_step(model: TinyNet) -> None:
    x = torch.randn(2, 4)
    y = torch.randint(0, 3, (2,))
    model.zero_grad(set_to_none=True)
    loss = nn.functional.cross_entropy(model(x), y)
    loss.backward()


def _make_session(epochs: int = 2, phases: dict[str, int] | None = None) -> tuple[Session, TinyNet]:
    if phases is None:
        phases = {"train": 2, "val": 2}
    model = TinyNet()
    return playgrad.start(model, epochs=epochs, phases=phases), model


def _run_in_thread(target) -> threading.Thread:
    thread = threading.Thread(target=target)
    thread.start()
    return thread


def test_run_mode_skips_capture_for_every_batch() -> None:
    session, model = _make_session()
    session.continue_run()

    captured: list[bool] = []
    for epoch in range(2):
        for phase, n in [("train", 2), ("val", 2)]:
            for _ in range(n):
                with session.batch(phase=phase, epoch=epoch) as ctx:
                    _train_step(model)
                captured.append(ctx.captured)

    assert captured == [False] * 8
    assert session.snapshot is None


def test_step_mode_pauses_on_every_batch() -> None:
    session, model = _make_session(epochs=1, phases={"train": 2})

    def loop() -> None:
        for _ in range(2):
            with session.batch(phase="train", epoch=0):
                _train_step(model)

    thread = _run_in_thread(loop)

    assert session.wait_until_paused(timeout=5)
    assert session.snapshot is not None
    assert session.snapshot.position.batch_idx == 0
    session.step_batch()

    assert session.wait_until_paused(after_pauses=1, timeout=5)
    assert session.snapshot is not None
    assert session.snapshot.position.batch_idx == 1
    session.continue_run()

    thread.join(timeout=5)
    assert not thread.is_alive()
    assert session.pause_count == 2


def test_until_phase_change_captures_only_phase_end() -> None:
    session, model = _make_session(epochs=1, phases={"train": 3, "val": 2})

    captured_positions: list[tuple[str, int, int]] = []

    def loop() -> None:
        for phase, n in [("train", 3), ("val", 2)]:
            for _ in range(n):
                with session.batch(phase=phase, epoch=0) as ctx:
                    _train_step(model)
                if ctx.captured and ctx.position is not None:
                    captured_positions.append(
                        (ctx.position.phase, ctx.position.epoch, ctx.position.batch_idx)
                    )

    session.step_phase()
    thread = _run_in_thread(loop)

    assert session.wait_until_paused(timeout=5)
    session.continue_run()

    thread.join(timeout=5)
    assert not thread.is_alive()
    assert captured_positions == [("train", 0, 2)]


def test_until_epoch_change_captures_only_epoch_end() -> None:
    session, model = _make_session(epochs=2, phases={"train": 2, "val": 2})

    captured_positions: list[tuple[str, int, int]] = []

    def loop() -> None:
        for epoch in range(2):
            for phase, n in [("train", 2), ("val", 2)]:
                for _ in range(n):
                    with session.batch(phase=phase, epoch=epoch) as ctx:
                        _train_step(model)
                    if ctx.captured and ctx.position is not None:
                        captured_positions.append(
                            (ctx.position.phase, ctx.position.epoch, ctx.position.batch_idx)
                        )

    session.step_epoch()
    thread = _run_in_thread(loop)

    assert session.wait_until_paused(timeout=5)
    session.continue_run()

    thread.join(timeout=5)
    assert not thread.is_alive()
    assert captured_positions == [("val", 0, 1)]


def test_snapshot_contains_activations_and_gradients() -> None:
    session, model = _make_session(epochs=1, phases={"train": 1})

    def loop() -> None:
        with session.batch(phase="train", epoch=0):
            _train_step(model)

    thread = _run_in_thread(loop)

    assert session.wait_until_paused(timeout=5)
    snap = session.snapshot
    assert snap is not None
    assert {"fc1", "fc2"} <= set(snap.activations)
    assert {"fc1.weight", "fc1.bias", "fc2.weight", "fc2.bias"} <= set(snap.gradients)
    expected_shapes = dict(model.named_parameters())
    for name, grad in snap.gradients.items():
        assert grad.shape == expected_shapes[name].shape

    session.continue_run()
    thread.join(timeout=5)


def test_stop_then_step_pauses_at_next_batch() -> None:
    session, model = _make_session(epochs=1, phases={"train": 3})
    session.continue_run()

    captured: list[bool] = []

    def loop() -> None:
        for _ in range(3):
            with session.batch(phase="train", epoch=0) as ctx:
                _train_step(model)
            captured.append(ctx.captured)

    thread = _run_in_thread(loop)
    session.stop()  # next batch boundary should pause
    assert session.wait_until_paused(timeout=5)
    session.continue_run()
    thread.join(timeout=5)
    assert not thread.is_alive()
    assert sum(captured) >= 1


def test_set_schedule_mid_run() -> None:
    session, model = _make_session(epochs=1, phases={"train": 2})
    session.continue_run()

    with session.batch(phase="train", epoch=0):
        _train_step(model)

    session.set_schedule(phases={"train": 5})
    for _ in range(4):
        with session.batch(phase="train", epoch=0):
            _train_step(model)


def test_close_releases_waiter_and_is_idempotent() -> None:
    session, model = _make_session(epochs=1, phases={"train": 1})

    def loop() -> None:
        with session.batch(phase="train", epoch=0):
            _train_step(model)

    thread = _run_in_thread(loop)
    assert session.wait_until_paused(timeout=5)
    session.close()
    session.close()
    thread.join(timeout=5)
    assert not thread.is_alive()
    assert session.closed


def test_close_before_any_batch_is_safe() -> None:
    session, model = _make_session(epochs=1, phases={"train": 1})
    session.close()
    with session.batch(phase="train", epoch=0) as ctx:
        _train_step(model)
    assert not ctx.captured
    assert session.snapshot is None


def test_unknown_phase_raises_through_context() -> None:
    session, model = _make_session(epochs=1, phases={"train": 1})
    session.continue_run()
    with pytest.raises(ValueError, match="unknown phase"):
        with session.batch(phase="bogus", epoch=0):
            _train_step(model)


def test_user_exception_does_not_pause() -> None:
    session, model = _make_session(epochs=1, phases={"train": 1})
    # default mode is STEP: would normally pause, but user exception should
    # propagate without us blocking the worker.

    class Boom(Exception):
        pass

    def loop() -> None:
        with pytest.raises(Boom):
            with session.batch(phase="train", epoch=0):
                raise Boom

    thread = _run_in_thread(loop)
    thread.join(timeout=5)
    assert not thread.is_alive()
    assert session.snapshot is None
    assert session.mode == Mode.STEP


def test_hooks_removed_after_each_batch() -> None:
    session, model = _make_session(epochs=1, phases={"train": 2})

    def loop() -> None:
        for _ in range(2):
            with session.batch(phase="train", epoch=0):
                _train_step(model)

    thread = _run_in_thread(loop)
    assert session.wait_until_paused(timeout=5)
    # Between pauses, hooks should have been removed even though we still hold
    # the activations from the previous batch on the snapshot.
    assert session._hook_handles == []  # type: ignore[reportPrivateUsage]
    session.continue_run()
    thread.join(timeout=5)
