"""The Kaggle notebook is valid JSON/ipynb and every code cell is valid Python."""

import ast
import json
from pathlib import Path

NB = Path(__file__).parent.parent / "notebooks" / "train_kaggle.ipynb"


def _code_cells():
    nb = json.loads(NB.read_text())
    assert nb["nbformat"] == 4
    return ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]


def _strip_magics(src: str) -> str:
    """Notebook shell lines (!pip ...) are not Python; replace them so ast can parse the rest."""
    return "\n".join(
        (line[: len(line) - len(line.lstrip())] + "pass")
        if line.lstrip().startswith(("!", "%"))
        else line
        for line in src.split("\n")
    )


def test_every_code_cell_parses():
    cells = _code_cells()
    assert len(cells) >= 6
    for src in cells:
        ast.parse(_strip_magics(src))


def test_notebook_uses_the_packed_data_and_the_right_install_flags():
    text = "\n".join(_code_cells())
    assert "--no-deps" in text and "--ignore-requires-python" in text  # don't clobber Kaggle torch
    assert "oscd.npz" in text and "sen1floods11.npz" in text
    assert "YOUR_GITHUB_USERNAME" in text  # placeholder the user must fill in
    assert "api_key" not in text.lower() and "kaggle.json" not in text  # no secrets in the notebook
