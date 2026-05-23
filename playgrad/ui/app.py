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

A `ui.timer` in each connection polls `session.pause_count`; when it
advances, the page re-renders all per-layer strips against
`session.snapshot`.
"""

from __future__ import annotations

import base64
import threading
from dataclasses import dataclass

import uvicorn
from fastapi import FastAPI
from nicegui import ui
from torch import nn

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
    layer_names = _hookable_module_names(session.model)

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


def _hookable_module_names(model: nn.Module) -> list[str]:
    return [name for name, m in model.named_modules() if m is not model]


@dataclass
class _PageState:
    sample_idx: int = 0
    last_pause_count: int = -1


def _build_page(
    session: Session,
    mermaid_src: str,
    layer_names: list[str],
) -> None:
    state = _PageState()
    layer_views: dict[str, _LayerView] = {}

    with ui.row().classes("w-full items-center gap-2 p-2 border-b"):
        ui.button("Stop", on_click=session.stop)
        ui.button("Step Batch", on_click=session.step_batch)
        ui.button("Step Phase", on_click=session.step_phase)
        ui.button("Step Epoch", on_click=session.step_epoch)
        ui.button("Step Run", on_click=session.step_run)
        ui.button("Detach", on_click=session.detach)
        position_label = ui.label("(waiting for first snapshot)").classes("ml-4 font-mono")
        ui.label("Sample:").classes("ml-4")
        sample_input = ui.number(value=0, min=0, step=1, format="%d").classes("w-20")

        def on_sample_change(e: object) -> None:
            value = getattr(e, "value", None)
            state.sample_idx = int(value) if value is not None else 0
            snap = session.snapshot
            if snap is not None:
                _render_all(state, layer_views, snap)

        sample_input.on_value_change(on_sample_change)

    with ui.row().classes("w-full no-wrap").style("height: calc(100vh - 64px)"):
        with ui.column().classes("w-1/2 h-full overflow-auto p-2"):
            ui.mermaid(mermaid_src).classes("w-full")
        with ui.column().classes("w-1/2 h-full overflow-auto p-2"):
            for name in layer_names:
                layer_views[name] = _LayerView(name)

    def tick() -> None:
        snap = session.snapshot
        if snap is None:
            return
        pos = snap.position
        position_label.text = (
            f"epoch {pos.epoch} | {pos.phase} batch {pos.batch_idx}"
            f" | pause {session.pause_count}"
        )
        if session.pause_count != state.last_pause_count:
            state.last_pause_count = session.pause_count
            _render_all(state, layer_views, snap)

    ui.timer(0.2, tick)


def _render_all(
    state: _PageState,
    views: dict[str, _LayerView],
    snap: BatchSnapshot,
) -> None:
    for name, view in views.items():
        view.update(
            snap.activations.get(name),
            snap.activation_gradients.get(name),
            state.sample_idx,
        )


class _LayerView:
    def __init__(self, name: str) -> None:
        self.name = name
        with ui.card().classes("w-full mb-2"):
            ui.label(name).classes("font-mono text-sm")
            with ui.element("div").classes("w-full overflow-x-auto"):
                self.act_image = ui.image().classes("max-w-none")
                self.grad_image = ui.image().classes("max-w-none")
        self.act_image.set_visibility(False)
        self.grad_image.set_visibility(False)

    def update(self, activation, gradient, sample_idx: int) -> None:
        act_png = render_strip(activation, sample_idx, kind="activation")
        grad_png = render_strip(gradient, sample_idx, kind="gradient")
        _set_or_hide(self.act_image, act_png)
        _set_or_hide(self.grad_image, grad_png)


def _set_or_hide(image: ui.image, png: bytes | None) -> None:
    if png is None:
        image.set_visibility(False)
        return
    image.set_source(_data_url(png))
    image.set_visibility(True)


def _data_url(png: bytes) -> str:
    return f"data:image/png;base64,{base64.b64encode(png).decode('ascii')}"
