# Playgrad internals

This document explains how the `playgrad` library is structured under the
hood. For *using* the library, see `README.md`. For agent-facing guidelines,
see `AGENTS.md`.

## Threading model

A playgrad session lives across two threads:

- **Training thread.** The user's training loop. Forward / backward / step
  run here, and `with session.batch(phase=..., epoch=...)` is entered here.
- **UI thread.** Driven by NiceGUI (not yet implemented). Reads session
  state, calls control methods (`stop`, `step_batch`, …, `detach`, `close`).

Synchronization is a single `threading.Condition` (`Session._cv`) protecting:

- `_mode` — current `Mode` enum value.
- `_resume_token` — monotonic counter; bumped by every "go" command.
- `_pause_count` — monotonic counter; bumped each time the training thread
  enters `_wait_for_proceed`.
- `_closed` — flips once on `close()`.
- `_schedule` — mutated by `set_schedule()` only.

`_snapshot`, `_activations`, and `_hook_handles` are written only by the
training thread and read by the UI thread; reads are point-in-time and
don't need a lock because Python attribute assignment is atomic under the
GIL.

## Schedule

A `Schedule` is constructed once at `playgrad.start(model, epochs=...,
phases=...)`. The `phases` dict is order-preserving (the last key in
insertion order is treated as the final phase of each epoch).

`Schedule.advance(phase, epoch)` is called inside `_BatchContext.__enter__`
and returns a `BatchPosition` with:

- `batch_idx` (0-based within `(phase, epoch)`)
- `is_last_in_phase`
- `is_last_in_epoch`
- `is_last_overall`

Because the run length is declared up-front, these flags are *predictive*:
we know on a batch's `__enter__` whether it is a boundary, before any
forward pass runs. That's what lets the session decide whether to install
hooks before the forward pass — there is no reactive "phase just changed"
detection at `__exit__`.

For non-deterministic workloads, `Session.set_schedule()` re-declares
`epochs` / `phases` mid-run.

## Modes and capture decisions

| Mode | Public method | Captures + pauses at |
|---|---|---|
| `STEP` | `step_batch()` (also `stop()`, no resume) | every batch |
| `UNTIL_PHASE_CHANGE` | `step_phase()` | `is_last_in_phase` |
| `UNTIL_EPOCH_CHANGE` | `step_epoch()` | `is_last_in_epoch` |
| `UNTIL_END` | `step_run()` | `is_last_overall` |
| `DETACH` | `detach()` | never |

A session starts in `STEP` mode — the first batch always pauses so the UI
can show its initial state.

`_should_capture(pos)` is the single decision function for both "install
hooks?" and "pause after this batch?". Capture and pause are intentionally
the same predicate — there is no implicit pause that the user did not ask
for, and there is no orphan capture without a pause to consume it.

## Hook lifecycle and gradient pickup

When `_should_capture` returns True, `__enter__` calls `_install_hooks()`,
which registers a forward hook on every submodule (skipping the top-level
model). The hook stores `output` in `Session._activations` as a *reference*
— no copy, no device transfer, no statistics. The activation tensors live
on the same device as the model.

At `__exit__`:

1. `_remove_hooks()` removes every registered hook.
2. If the batch ran without an exception and the session isn't closed,
   `_publish_snapshot()` builds a `BatchSnapshot`:
   - `activations`: a shallow copy of the hook-stored dict.
   - `gradients`: `{name: param.grad for name, param in
     model.named_parameters() if param.grad is not None}`.
3. The training thread calls `_wait_for_proceed()` (described below).
4. After resume, `_activations` is cleared so the next batch starts clean.

Gradients are read straight off `param.grad` rather than via backward
hooks. This works because the user's training loop calls
`optimizer.zero_grad()` at the *start* of their batch body, before
`loss.backward()`. By `__exit__`, the gradients from this batch are still
on the parameters. The session does not need to register backward hooks
or copy grads into a buffer — it reads them at snapshot time and the live
tensors get reused (or zeroed) by the user's next batch.

The trade-off: activation-gradients (∂loss/∂activation) are not captured.
Weight gradients (the ones that actually update parameters) are.

## Resume mechanism

`_wait_for_proceed()` uses a token-counter pattern instead of an
`Event`:

```python
def _wait_for_proceed(self) -> None:
    with self._cv:
        seen = self._resume_token
        self._pause_count += 1
        self._cv.notify_all()
        while self._resume_token == seen and not self._closed:
            self._cv.wait()
```

Each `step_*` / `detach()` call bumps `_resume_token` under the lock and
`notify_all`s. The waiting thread loops until the token has advanced past
the value it captured on entry.

Why not a plain `Event`? `Event.set()` is idempotent: if the UI sends two
rapid commands (e.g. `step_batch()` followed immediately by `detach()`)
while the worker is in flight, only one set survives `wait/clear`, so the
next pause would deadlock. The token pattern is robust to coalesced
commands — each pause only requires *any* resume command issued after the
pause began.

`_pause_count` is the symmetric counter for the UI side:
`wait_until_paused(after_pauses=N, timeout=...)` blocks until the worker
has paused more than `N` times. Tests and the UI use this to synchronize
without polling.

## Snapshot lifecycle

`Session.snapshot` is the last `BatchSnapshot` published, or `None` if no
batch has been captured yet. It persists after `close()`, so the UI can
stay open and present a post-mortem view.

The snapshot holds *references* to tensors that live on the model's device.
The UI is expected to compute statistics / downsample / render lazily and
on-device, only producing CPU bytes when the user opens a specific layer's
view. As a consequence: while the worker is paused, the activations are
pinned in GPU memory; resuming releases them at the next `__exit__` cleanup
step.

## Lifecycle summary

```text
playgrad.start(model, epochs, phases)
        │
        ▼
   Session (mode=STEP)
        │
        ├── with session.batch(phase, epoch):
        │       ▼
        │   schedule.advance() → BatchPosition
        │   _should_capture()? ── no ──┐
        │       │ yes                   │
        │       ▼                       │
        │   _install_hooks()            │
        │       │                       │
        │       (user code:             │
        │        zero_grad, forward,    │
        │        backward, step)        │
        │       │                       │
        │       ▼                       │
        │   _remove_hooks()             │
        │   _publish_snapshot()         │
        │   _wait_for_proceed() ────────┤
        │       │                       │
        │       ▼                       ▼
        │   _activations.clear()    (no capture, no pause)
        │
        ▼ (UI thread, anytime)
   stop / step_batch / step_phase / step_epoch /
   step_run / detach / close
```
