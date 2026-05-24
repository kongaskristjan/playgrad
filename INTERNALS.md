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
| `UNTIL_POSITION` | `step_until_position(phase, epoch, batch_idx)` | exactly that `(phase, epoch, batch_idx)` |
| `DETACH` | `detach()` | never |

A session starts in `STEP` mode — the first batch always pauses so the UI
can show its initial state.

`_should_capture(pos)` is the single decision function for both "install
hooks?" and "pause after this batch?". Capture and pause are intentionally
the same predicate — there is no implicit pause that the user did not ask
for, and there is no orphan capture without a pause to consume it.

## Hook lifecycle, gradient pickup, snapshot copy

There are two capture paths, picked once at session construction by
trying `torch.fx.symbolic_trace(model)`.

**fx path (preferred).** When the trace succeeds, the session holds the
resulting `fx.GraphModule` and `_install_hooks` monkey-patches
`model.forward` with a function that runs a custom `fx.Interpreter`
subclass against that graph. The interpreter overrides `run_node` so that
after every node executes — placeholders, `call_module`, `call_function`,
`call_method` — it stores the returned tensor in `Session._activations`
under a friendly key (the dotted target for module calls, the fx node
name for everything else: `x`, `relu`, `relu_1`, `add`). Each captured
tensor gets `retain_grad()` so the user's `loss.backward()` populates
`.grad`. This is what lets `torch.relu(...)`, `out + shortcut(x)`, and
similar non-module operations show up in the UI on equal footing with
named modules. `_remove_hooks` restores the original `forward`.

**Hook fallback.** When `fx.symbolic_trace` raises (data-dependent
control flow, tracing-unfriendly ops, etc.), the session falls back to:

1. A forward **pre-hook** on the root model. It captures each positional
   tensor input under its parameter name (derived from
   `inspect.signature(model.forward)`), e.g. `x` for a model whose
   forward signature is `forward(self, x)`.
2. A forward hook on every submodule (skipping the root model itself).
   The hook stores `output` in `Session._activations` as a live
   reference *and* calls `output.retain_grad()` on it (when
   `requires_grad`) so that PyTorch populates `output.grad` after
   `loss.backward()`. No backward hook is needed.

In both paths `Session.layer_names` is computed once at construction and
exposes the same key set the UI later reads from each `BatchSnapshot`.

At `__exit__`:

1. `_remove_hooks()` removes every registered hook.
2. If the batch ran without an exception and the session isn't closed,
   `_publish_snapshot()` builds a `BatchSnapshot` containing **CPU clones**
   of four tensor dicts:
   - `activations`: hook-stored module outputs, cloned to CPU.
   - `activation_gradients`: `activation.grad` for each captured output
     that has one, cloned to CPU.
   - `weights`: every `param` from `named_parameters()`, cloned to CPU.
   - `weight_gradients`: `param.grad` where non-`None`, cloned to CPU.
   Every clone goes through `tensor.detach().to("cpu", copy=True)`, so the
   snapshot is fully independent of the live computation graph — the next
   batch can free / overwrite all of its source tensors without affecting
   the snapshot.
3. The training thread calls `_wait_for_proceed()` (described below).
4. After resume, `_activations` is cleared so the next batch starts clean.

Weight gradients are read straight off `param.grad` rather than via
backward hooks. This works because the user's training loop calls
`optimizer.zero_grad()` at the *start* of their batch body, before
`loss.backward()`. By `__exit__`, the gradients from this batch are still
on the parameters; `optimizer.step()` does not touch `.grad`.

Memory profile: the eager CPU clone costs O(activations + parameters)
bytes of host memory per captured batch. For small models (ResNet-20)
this is tens of MB; for large models, a `watch=` filter to opt out of
some modules is the future escape hatch. In exchange, the snapshot is
thread-safe to read from the UI without holding any session lock and
survives arbitrarily long after the training thread has moved on.

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

The snapshot is a frozen dataclass of CPU tensors. Assignment is a single
attribute write; readers in the UI thread observe either the previous
snapshot or the new one — never a torn half-written state. The UI can
hold references to a snapshot for as long as it wants without preventing
the next batch from running.

Rendering (PNG mosaics, histograms, summary stats) is still computed
lazily on the UI thread when a layer card is opened; the eager copy in
`_publish_snapshot` only moves raw tensor data, not anything pixel-shaped.
A `(layer, pause_count)` cache in the UI keeps re-opens free.

## UI layer

`playgrad.ui` is a thin NiceGUI app that reads `Session.snapshot` and
drives `Session` via the five control methods plus `detach` and `close`.
It does not touch tensors directly until they need to be rendered.

- `playgrad.ui.graph.build_mermaid(model)` produces the Mermaid TD source
  for the architecture view. It tries `torch.fx.symbolic_trace(model)`
  first, which yields a real data-flow graph — vertical chains, with
  branches and merges at residual blocks. For models that aren't
  fx-traceable (dynamic control flow, custom ops), it falls back to a
  static module-hierarchy tree rooted at a synthetic `root` node. Nodes
  use different Mermaid shapes per fx op (rectangles for `call_module`,
  circles for `call_function` / `call_method`, stadiums for
  `placeholder` / `output`).
- `playgrad.ui.render.render_strip(tensor, sample_idx, kind=...)` is the
  function that turns per-layer CPU tensors into PNG bytes:
  - For per-sample shape `[C, H, W]` it interpolates each channel to a
    `TILE_SIZE × TILE_SIZE` tile and concatenates horizontally with a
    `TILE_GAP`-px white spacer between tiles so adjacent dark channels
    don't smear together.
  - For `[F]` it builds a single short heatmap row, downsampled to at
    most `LINEAR_MAX_BINS` bins when `F` is large.
  - Sequential grayscale colormap for activations, diverging
    blue-white-red for gradients. PNG `compress_level=1` — wire size
    doesn't matter, encode speed does.
  - Other per-sample shapes return `None`; the UI hides those images.
- `playgrad.ui.render.render_image(tensor, sample_idx, mean=..., std=...)`
  renders the model input as a natural RGB or grayscale PNG (upscaled to
  `INPUT_IMAGE_SIZE` with nearest-neighbour). Channels are assumed to lie
  in `[0, 1]` unless both `mean` and `std` are passed, in which case the
  sample is denormalized (`x * std + mean`) before being clamped and
  scaled to 8-bit. Anything other than `C == 1` or `C == 3` returns
  `None`.
- `playgrad.ui.app.serve(session, port=..., host=...)` runs the NiceGUI
  app on a background thread. NiceGUI is mounted onto a bare FastAPI
  app via `ui.run_with`, which is then served by `uvicorn.Server` from
  the thread. `install_signal_handlers` is patched to a no-op because
  uvicorn would otherwise try to register SIGINT/SIGTERM handlers from
  a non-main thread. The thread is non-daemon, so the UI stays alive
  even after the training script's main thread returns — the user
  closes the browser / Ctrl-Cs when they're done browsing post-mortem.
- The page handler creates one `_LayerView` per submodule (a card with
  two `ui.image` strips inside a shared horizontal scroll container)
  and a `ui.timer` that, every 200 ms, checks `session.pause_count`. If
  it has advanced since the last render, every layer view re-renders
  against the new snapshot, slicing each tensor at the current
  `sample_idx` (driven by a single `ui.number` input in the top bar).
- Rendering is intentionally eager when a new snapshot lands — for
  ResNet-20 it takes well under a second, and during `RUN` / `DETACH`
  modes no snapshots are produced so the UI is idle. For larger models
  this is the natural point to add viewport-aware lazy rendering, but
  the current code path keeps the wiring simple.

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
   step_run / step_until_position / detach / close
```
