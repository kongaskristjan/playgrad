"""Tests for the Schedule / BatchPosition machinery."""

from __future__ import annotations

import pytest

from playgrad.schedule import Schedule


def test_advance_through_full_run() -> None:
    schedule = Schedule(epochs=2, phases={"train": 3, "val": 2})

    positions = []
    for epoch in range(2):
        for phase, n in [("train", 3), ("val", 2)]:
            for _ in range(n):
                positions.append(schedule.advance(phase, epoch))

    assert [(p.phase, p.epoch, p.batch_idx) for p in positions] == [
        ("train", 0, 0), ("train", 0, 1), ("train", 0, 2),
        ("val", 0, 0), ("val", 0, 1),
        ("train", 1, 0), ("train", 1, 1), ("train", 1, 2),
        ("val", 1, 0), ("val", 1, 1),
    ]
    last_in_phase = [p.is_last_in_phase for p in positions]
    assert last_in_phase == [
        False, False, True,
        False, True,
        False, False, True,
        False, True,
    ]
    assert sum(p.is_last_in_epoch for p in positions) == 2
    assert sum(p.is_last_overall for p in positions) == 1
    assert positions[-1].is_last_overall


@pytest.mark.parametrize(
    ("epochs", "phases"),
    [
        (0, {"train": 1}),
        (-1, {"train": 1}),
        (1, {}),
        (1, {"train": 0}),
        (1, {"train": -2}),
    ],
)
def test_invalid_construction(epochs: int, phases: dict[str, int]) -> None:
    with pytest.raises(ValueError):
        Schedule(epochs=epochs, phases=phases)


def test_unknown_phase_raises() -> None:
    schedule = Schedule(epochs=1, phases={"train": 2})
    with pytest.raises(ValueError, match="unknown phase"):
        schedule.advance("val", 0)


def test_out_of_range_epoch_raises() -> None:
    schedule = Schedule(epochs=2, phases={"train": 2})
    with pytest.raises(ValueError, match="out of range"):
        schedule.advance("train", 2)


def test_too_many_batches_raises() -> None:
    schedule = Schedule(epochs=1, phases={"train": 2})
    schedule.advance("train", 0)
    schedule.advance("train", 0)
    with pytest.raises(ValueError, match="more batches than declared"):
        schedule.advance("train", 0)


def test_update_changes_counts() -> None:
    schedule = Schedule(epochs=1, phases={"train": 2})
    schedule.update(phases={"train": 5})
    assert schedule.phases == {"train": 5}
    schedule.advance("train", 0)
    schedule.advance("train", 0)
    schedule.advance("train", 0)


def test_last_phase_name_follows_insertion_order() -> None:
    schedule = Schedule(epochs=1, phases={"train": 1, "val": 1})
    assert schedule.last_phase_name == "val"
    schedule.update(phases={"val": 1, "train": 1})
    assert schedule.last_phase_name == "train"
