"""Build a Mermaid TD diagram source from a model's module hierarchy."""

from __future__ import annotations

import re

from torch import nn

CONFIG_HEADER: str = """---
config:
  layout: elk
  theme: neutral
  look: neo
  elk:
    mergeEdges: true
---
"""

ROOT_ID: str = "root"


def build_mermaid(model: nn.Module, *, root_label: str = "model") -> str:
    """Return Mermaid source for the model's module hierarchy.

    Each submodule becomes a node labelled `<full_name><br/><class>`; edges
    run from parent to direct children, so containers like `nn.Sequential`
    appear with their children fanned out underneath.
    """
    lines: list[str] = [CONFIG_HEADER, "flowchart TD", f'  {ROOT_ID}["{root_label}"]']

    for full_name, module in model.named_modules():
        if module is model:
            continue
        node_id = _slug(full_name)
        label = f"{full_name}<br/>{type(module).__name__}"
        lines.append(f'  {node_id}["{label}"]')

    for full_name, module in model.named_modules():
        parent_id = ROOT_ID if module is model else _slug(full_name)
        for child_name, _ in module.named_children():
            child_full = f"{full_name}.{child_name}" if full_name else child_name
            lines.append(f"  {parent_id} --> {_slug(child_full)}")

    return "\n".join(lines)


def _slug(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9]", "_", name) or ROOT_ID
