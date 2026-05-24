"""NiceGUI app that visualizes a `Session`.

The app runs on a background daemon-thread-disabled uvicorn (signal handlers
disabled so it survives being started from a non-main thread). Layout:

- Top bar with the six control buttons, a position label, and a sample
  index spinner.
- Left pane: the model architecture as a Mermaid diagram (built once at
  start).
- Right pane: one card per submodule. Each card holds two horizontally
  scrollable strips — activations on top, activation gradients below —
  sharing a single horizontal scrollbar so they pan together.

A `ui.timer` in each connection polls `session.snapshot`; when a new
snapshot is published, the page re-renders all per-layer strips against
it.
"""

from __future__ import annotations

import asyncio
import base64
import threading
from dataclasses import dataclass

import uvicorn
from fastapi import FastAPI
from nicegui import ui
from torch import Tensor

from playgrad.schedule import Schedule
from playgrad.session import BatchSnapshot, Session
from playgrad.ui.graph import build_mermaid
from playgrad.ui.render import render_strip


def serve(
    session: Session,
    *,
    port: int = 8080,
    host: str = "127.0.0.1",
    log_level: str = "warning",
) -> threading.Thread:
    """Start the NiceGUI app on a background thread and return that thread.

    NiceGUI is mounted onto a bare FastAPI app via `ui.run_with`; the app is
    then served by uvicorn from a non-main thread, with signal handlers
    disabled so uvicorn doesn't try to wire SIGINT/SIGTERM from a thread
    that isn't the main one.
    """
    mermaid_src = build_mermaid(session.model)
    layer_names = session.layer_names

    fastapi_app = FastAPI()

    @ui.page("/")
    def index() -> None:
        _build_page(session, mermaid_src, layer_names)

    ui.run_with(fastapi_app, storage_secret="playgrad")

    config = uvicorn.Config(
        app=fastapi_app,
        host=host,
        port=port,
        log_level=log_level,
    )
    server = uvicorn.Server(config)
    setattr(server, "install_signal_handlers", lambda: None)

    thread = threading.Thread(target=server.run, name="playgrad-ui", daemon=False)
    thread.start()
    return thread


@dataclass
class _PageState:
    sample_idx: int = 0
    last_snapshot: BatchSnapshot | None = None
    dirty: bool = False
    rendering: bool = False
    spinner_max: int | None = None


def _build_page(
    session: Session,
    mermaid_src: str,
    layer_names: list[str],
) -> None:
    state = _PageState()
    layer_views: dict[str, _LayerView] = {}

    step_until_custom = _build_step_until_custom_dialog(session)

    with ui.row().classes("w-full items-center gap-2 p-2 border-b"):
        ui.button("Stop", on_click=session.stop)
        ui.button("Step Batch", on_click=session.step_batch)
        ui.button("Step Epoch", on_click=session.step_epoch)
        ui.button("Step Until End", on_click=session.step_run)
        ui.button("Step Until Custom", on_click=step_until_custom.open)
        ui.button("Detach", on_click=session.detach)
        position_label = ui.label("(waiting for first snapshot)").classes("ml-4 font-mono")
        ui.label("Sample:").classes("ml-4")
        sample_input = ui.number(value=0, min=0, step=1, format="%d").classes("w-20")

        def _defer_clamp_display(target: int) -> None:
            # NiceGUI suppresses .value writes made from inside a value-change
            # handler; schedule the display correction for the next event loop
            # iteration so it actually reaches the client.
            ui.timer(0.0, lambda: sample_input.set_value(target), once=True)

        def on_sample_change(e: object) -> None:
            value = getattr(e, "value", None)
            idx = int(value) if value is not None else 0
            if idx < 0:
                idx = 0
                _defer_clamp_display(idx)
            elif state.spinner_max is not None and idx > state.spinner_max:
                idx = state.spinner_max
                _defer_clamp_display(idx)
            state.sample_idx = idx
            state.dirty = True

        sample_input.on_value_change(on_sample_change)

    with ui.row().classes("w-full no-wrap").style("height: calc(100vh - 64px)"):
        with ui.column().classes("w-1/2 h-full overflow-auto p-2"):
            ui.mermaid(mermaid_src).classes("w-full")
        with ui.column().classes("w-1/2 h-full overflow-auto p-2"):
            for name in layer_names:
                layer_views[name] = _LayerView(name)

    async def tick() -> None:
        snap = session.snapshot
        if snap is None:
            return
        pos = snap.position
        position_label.text = f"epoch {pos.epoch} | {pos.phase} batch {pos.batch_idx}"
        _sync_spinner_max(snap, state, sample_input)
        if state.rendering:
            return
        if snap is not state.last_snapshot or state.dirty:
            state.last_snapshot = snap
            state.dirty = False
            state.rendering = True
            try:
                sample_idx = state.sample_idx
                rendered = await asyncio.to_thread(
                    _compute_all, layer_views, snap, sample_idx
                )
            finally:
                state.rendering = False
            _apply_all(layer_views, rendered)

    ui.timer(0.2, tick)


def _build_step_until_custom_dialog(session: Session) -> ui.dialog:
    schedule = session.schedule
    phase_names = list(schedule.phases)

    with ui.dialog() as dialog, ui.card():
        ui.label("Step until position").classes("text-lg font-bold")
        epoch_input = ui.number(
            label="Epoch", value=0, min=0, step=1, format="%d"
        ).classes("w-32")
        phase_select = ui.select(
            phase_names, label="Phase", value=phase_names[0]
        ).classes("w-32")
        batch_input = ui.number(
            label="Batch", value=0, min=0, step=1, format="%d"
        ).classes("w-32")
        error_label = ui.label("").classes("text-red-500 text-sm min-h-4")

        def submit() -> None:
            try:
                epoch = int(epoch_input.value) if epoch_input.value is not None else 0
                batch_idx = int(batch_input.value) if batch_input.value is not None else 0
            except (TypeError, ValueError):
                error_label.text = "Invalid input"
                return
            phase = str(phase_select.value)
            error = _validate_step_until_target(
                schedule=schedule,
                snapshot=session.snapshot,
                phase=phase,
                epoch=epoch,
                batch_idx=batch_idx,
            )
            if error is not None:
                error_label.text = error
                return
            error_label.text = ""
            session.step_until_position(phase=phase, epoch=epoch, batch_idx=batch_idx)
            dialog.close()

        with ui.row():
            ui.button("Cancel", on_click=dialog.close)
            ui.button("Step", on_click=submit)

    return dialog


def _validate_step_until_target(
    *,
    schedule: Schedule,
    snapshot: BatchSnapshot | None,
    phase: str,
    epoch: int,
    batch_idx: int,
) -> str | None:
    phases = schedule.phases
    if phase not in phases:
        return f"Unknown phase {phase!r}"
    if not 0 <= epoch < schedule.epochs:
        return f"Epoch must be in [0, {schedule.epochs - 1}]"
    declared = phases[phase]
    if not 0 <= batch_idx < declared:
        return f"Batch must be in [0, {declared - 1}] for phase {phase!r}"
    if snapshot is not None:
        cur = snapshot.position
        target_rank = _position_rank(phases, phase, epoch, batch_idx)
        current_rank = _position_rank(phases, cur.phase, cur.epoch, cur.batch_idx)
        if target_rank <= current_rank:
            return "Target must be after the current position"
    return None


def _position_rank(
    phases: dict[str, int], phase: str, epoch: int, batch_idx: int
) -> tuple[int, int, int]:
    return (epoch, list(phases).index(phase), batch_idx)


def _snapshot_batch_size(snap: BatchSnapshot) -> int | None:
    for tensor in snap.activations.values():
        if tensor.ndim > 0:
            return int(tensor.shape[0])
    return None


def _sync_spinner_max(
    snap: BatchSnapshot,
    state: _PageState,
    sample_input: ui.number,
) -> None:
    batch_size = _snapshot_batch_size(snap)
    if batch_size is None or batch_size <= 0:
        return
    new_max = batch_size - 1
    if state.spinner_max == new_max:
        return
    state.spinner_max = new_max
    sample_input.props(f"max={new_max}")
    if state.sample_idx > new_max:
        state.sample_idx = new_max
        sample_input.value = new_max
        state.dirty = True


def _compute_all(
    views: dict[str, _LayerView],
    snap: BatchSnapshot,
    sample_idx: int,
) -> dict[str, tuple[str, str]]:
    return {
        name: view.compute(
            snap.activations.get(name),
            snap.activation_gradients.get(name),
            sample_idx,
        )
        for name, view in views.items()
    }


def _apply_all(
    views: dict[str, _LayerView],
    rendered: dict[str, tuple[str, str]],
) -> None:
    for name, (act_html, grad_html) in rendered.items():
        views[name].apply(act_html, grad_html)


class _LayerView:
    """One card per submodule, with activation + activation-gradient strips.

    The strips are raw `<img>` elements with `max-width: none`, so each PNG
    renders at its natural pixel width and the wrapping `overflow-x-auto`
    div produces a shared horizontal scrollbar. NiceGUI's `ui.image` uses
    Quasar's responsive q-img instead, which squishes the strip to the
    card width — not what we want here.
    """

    def __init__(self, name: str) -> None:
        self.name = name
        with ui.card().classes("w-full mb-2"):
            ui.label(name).classes("font-mono text-sm")
            with ui.element("div").classes("w-full overflow-x-auto"):
                self.act_html = ui.html("")
                self.grad_html = ui.html("")

    def compute(
        self, activation: Tensor | None, gradient: Tensor | None, sample_idx: int
    ) -> tuple[str, str]:
        act_png = render_strip(activation, sample_idx, kind="activation")
        grad_png = render_strip(gradient, sample_idx, kind="gradient")
        return _img_tag(act_png), _img_tag(grad_png)

    def apply(self, act_html: str, grad_html: str) -> None:
        self.act_html.set_content(act_html)
        self.grad_html.set_content(grad_html)


def _img_tag(png: bytes | None) -> str:
    if png is None:
        return ""
    b64 = base64.b64encode(png).decode("ascii")
    return (
        f'<img src="data:image/png;base64,{b64}" '
        'style="display:block; max-width:none;" />'
    )
