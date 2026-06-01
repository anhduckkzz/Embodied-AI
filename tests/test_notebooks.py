"""Structural checks for the curriculum notebooks.

The notebooks are interactive learning artifacts, so these tests do not execute
heavy cells. They verify that notebooks stay valid, keep the required study
scaffold, and avoid obvious JSON/control-character corruption.
"""
from __future__ import annotations

import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = sorted((ROOT / "curriculum").rglob("notebook.ipynb"))
REQUIRED_MARKERS = [
    "## Notebook type and ordered study structure",
    "## From-scratch focus and code-reading checklist",
    "## Visualization and debugging ideas",
    "## Scratch → framework mapping",
    "## Real-system application",
    "## Failure modes and debugging",
    "## Mini-project and mastery checklist",
]


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_all_curriculum_notebooks_are_present_and_valid_json():
    assert len(NOTEBOOKS) == 40
    for path in NOTEBOOKS:
        nb = _load(path)
        assert nb["nbformat"] == 4, path
        assert isinstance(nb["cells"], list) and nb["cells"], path


def test_notebooks_keep_required_beginner_study_scaffold():
    for path in NOTEBOOKS:
        text = "\n".join("".join(cell.get("source", [])) for cell in _load(path)["cells"])
        for marker in REQUIRED_MARKERS:
            assert marker in text, f"{path} missing {marker}"
        assert "Type A" in text or "Type B" in text or "Type C" in text, path


def test_notebooks_do_not_contain_non_newline_control_characters():
    for path in NOTEBOOKS:
        text = path.read_text(encoding="utf-8")
        bad = [ch for ch in text if ord(ch) < 32 and ch not in "\n\r\t"]
        assert not bad, path


def test_python_code_cells_parse_after_ignoring_notebook_magics():
    for path in NOTEBOOKS:
        nb = _load(path)
        for i, cell in enumerate(nb["cells"]):
            if cell.get("cell_type") != "code":
                continue
            source = "".join(cell.get("source", []))
            python_lines = []
            for line in source.splitlines():
                stripped = line.lstrip()
                if stripped.startswith("!") or stripped.startswith("%"):
                    # Preserve indentation so an indented notebook shell command
                    # inside an if/for block still leaves syntactically valid Python.
                    indent = line[: len(line) - len(stripped)]
                    python_lines.append(indent + "pass")
                    continue
                python_lines.append(line)
            filtered = "\n".join(python_lines).strip()
            if filtered:
                ast.parse(filtered, filename=f"{path}:cell-{i}")
