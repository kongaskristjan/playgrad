"""NiceGUI app that visualizes a `Session`.

The app runs on a background daemon-thread-disabled uvicorn (signal handlers
disabled so it survives being started from a non-main thread). Layout:

- Top bar with the six control buttons, a position label, and a sample
  index spinner.
- Left pane: the model architecture as a Mermaid diagram (built once at
  start).
- Right pane: a one-column table with one row per submodule. Each row
  holds two horizontally scrollable strips — activations on top,
  activation gradients below — sharing a single horizontal scrollbar so
  they pan together.

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
from playgrad.ui.render import render_image, render_strip


def serve(
    session: Session,
    *,
    port: int = 8080,
    host: str = "127.0.0.1",
    log_level: str = "warning",
    input_mean: tuple[float, ...] | None = None,
    input_std: tuple[float, ...] | None = None,
) -> threading.Thread:
    """Start the NiceGUI app on a background thread and return that thread.

    NiceGUI is mounted onto a bare FastAPI app via `ui.run_with`; the app is
    then served by uvicorn from a non-main thread, with signal handlers
    disabled so uvicorn doesn't try to wire SIGINT/SIGTERM from a thread
    that isn't the main one.

    `input_mean` / `input_std` are passed to the input-image pane so the
    sample is denormalized (`x * std + mean`) before display. When either
    is `None`, the renderer assumes the input is already in `[0, 1]`.
    """
    mermaid_src = build_mermaid(session.model)
    layer_names = session.layer_names
    input_name = session.input_names[0] if session.input_names else None

    fastapi_app = FastAPI()

    @ui.page("/")
    def index() -> None:
        _build_page(
            session,
            mermaid_src,
            layer_names,
            input_name=input_name,
            input_mean=input_mean,
            input_std=input_std,
        )

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
    *,
    input_name: str | None,
    input_mean: tuple[float, ...] | None,
    input_std: tuple[float, ...] | None,
) -> None:
    state = _PageState()
    layer_views: dict[str, _LayerView] = {}

    ui.query(".nicegui-content").classes("p-0 h-screen overflow-hidden")
    ui.query("body").classes("overflow-hidden")
    ui.query("html").classes("overflow-hidden")

    step_until_custom = _build_step_until_custom_dialog(session)

    with ui.column().classes("w-full h-screen no-wrap gap-0"):
        with ui.row().classes(
            "w-full items-center gap-x-3 gap-y-0 px-3 py-2 shrink-0 "
            "border-b-2 border-slate-300 bg-slate-100 shadow-sm z-10"
        ):
            architecture_toggle = ui.button(
                icon="account_tree", color="slate-500"
            ).props("dense size=md").tooltip("Toggle architecture pane")
            ui.button("Stop", on_click=session.stop, color="red").props("dense size=md")
            ui.button("Step Batch", on_click=session.step_batch, color="orange").props("dense size=md")
            ui.button("Step Epoch", on_click=session.step_epoch, color="orange").props("dense size=md")
            ui.button("Step Until End", on_click=session.step_run, color="orange").props("dense size=md")
            ui.button("Step Until Custom", on_click=step_until_custom.open, color="orange").props("dense size=md")
            ui.button("Detach", on_click=session.detach, color="green").props("dense size=md")
            position_label = ui.label("(waiting for first snapshot)").classes("ml-3 font-mono text-sm")
            ui.label("Sample:").classes("ml-3 text-sm")
            sample_input = ui.number(value=0, min=0, step=1, format="%d").classes("w-20").props("dense")
            input_toggle = ui.button(
                icon="image", color="slate-500"
            ).classes("ml-auto").props("dense size=md").tooltip(
                "Toggle input image pane"
            )

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

        with ui.row().classes("w-full no-wrap gap-0 grow min-h-0"):
            architecture_pane = ui.column().classes(
                "w-1/4 shrink-0 h-full overflow-auto p-2 "
                "border-r-2 border-slate-300 bg-slate-50"
            )
            with architecture_pane:
                ui.mermaid(mermaid_src).classes("w-full")
            with ui.column().classes(
                "grow min-w-0 h-full overflow-auto p-3 bg-slate-200 gap-3"
            ):
                for name in layer_names:
                    layer_views[name] = _LayerView(name)
            input_pane = ui.column().classes(
                "w-72 shrink-0 h-full overflow-auto p-3 "
                "border-l-2 border-slate-300 bg-slate-50 items-center"
            )
            with input_pane:
                ui.label("Input").classes("font-mono text-sm self-start")
                input_html = ui.html("")

        def toggle_architecture() -> None:
            architecture_pane.set_visibility(not architecture_pane.visible)

        def toggle_input() -> None:
            input_pane.set_visibility(not input_pane.visible)

        architecture_toggle.on_click(toggle_architecture)
        input_toggle.on_click(toggle_input)

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
                rendered, input_img = await asyncio.to_thread(
                    _compute_frame,
                    layer_views,
                    snap,
                    sample_idx,
                    input_name=input_name,
                    input_mean=input_mean,
                    input_std=input_std,
                )
            finally:
                state.rendering = False
            _apply_all(layer_views, rendered)
            input_html.set_content(input_img)

    ui.timer(0.2, tick)


def _build_step_until_custom_dialog(session: Session) -> ui.dialog:
    schedule = session.schedule
    phase_names = list(schedule.phases)

    with ui.dialog() as dialog, ui.card().classes("min-w-96 p-6 gap-4"):
        ui.label("Step until custom").classes("text-lg font-bold")
        with ui.row().classes("w-full gap-4 items-end no-wrap"):
            epoch_input = ui.number(
                label="Epoch", value=0, min=0, step=1, format="%d"
            ).classes("flex-1")
            phase_select = ui.select(
                phase_names, label="Phase", value=phase_names[0]
            ).classes("flex-1")
            batch_input = ui.number(
                label="Batch", value=0, min=0, step=1, format="%d"
            ).classes("flex-1")
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


def _compute_frame(
    views: dict[str, _LayerView],
    snap: BatchSnapshot,
    sample_idx: int,
    *,
    input_name: str | None,
    input_mean: tuple[float, ...] | None,
    input_std: tuple[float, ...] | None,
) -> tuple[dict[str, tuple[str, str]], str]:
    rendered = {
        name: view.compute(
            snap.activations.get(name),
            snap.activation_gradients.get(name),
            sample_idx,
        )
        for name, view in views.items()
    }
    input_tensor = snap.activations.get(input_name) if input_name else None
    input_png = render_image(
        input_tensor, sample_idx, mean=input_mean, std=input_std
    )
    return rendered, _img_tag(input_png)


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
    div produces a shared horizontal scrollbar inside the card. NiceGUI's
    `ui.image` uses Quasar's responsive q-img instead, which squishes the
    strip to the card width — not what we want here. The card has
    `min-w-0` so a wide strip doesn't push the column wider.
    """

    def __init__(self, name: str) -> None:
        self.name = name
        with ui.element("div").classes(
            "w-full min-w-0 bg-white rounded border border-slate-300 shadow-sm "
            "hover:border-blue-400"
        ):
            ui.label(name).classes(
                "block px-3 py-1 font-mono text-sm bg-slate-100 "
                "border-b border-slate-300 rounded-t"
            )
            with ui.element("div").classes("w-full overflow-x-auto p-2"):
                self.act_html = ui.html("")
                ui.element("div").classes("h-1")
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
