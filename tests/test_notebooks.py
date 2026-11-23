import ast
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).parents[1]
NOTEBOOKS = sorted((ROOT / "notebooks").glob("*.ipynb"))


def test_all_notebooks_have_current_executable_python_cells():
    assert len(NOTEBOOKS) == 27
    for path in NOTEBOOKS:
        notebook = json.loads(path.read_text(encoding="utf-8"))
        assert notebook["nbformat"] == 4
        code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
        assert code_cells
        for cell in code_cells:
            ast.parse(cell["source"])
            assert cell["outputs"] == []
            assert cell["execution_count"] is None
        code_text = "\n".join(cell["source"] for cell in code_cells)
        for legacy_import in (
            "import pandas",
            "from pandas",
            "import PyFin",
            "from PyFin",
            "import cvxopt",
            "from cvxopt",
        ):
            assert legacy_import not in code_text, (path.name, legacy_import)


def test_notebook_index_covers_every_notebook():
    index = (ROOT / "notebooks" / "README.md").read_text(encoding="utf-8")
    for path in NOTEBOOKS:
        assert f"<{path.name}>" in index


def test_original_notebook_markdown_and_code_comments_are_preserved():
    for path in NOTEBOOKS:
        original = json.loads(
            subprocess.check_output(["git", "show", f"7b7361c:notebooks/{path.name}"], text=True)
        )
        current = json.loads(path.read_text(encoding="utf-8"))
        original_markdown = [
            "".join(cell["source"]) for cell in original["cells"] if cell["cell_type"] == "markdown"
        ]
        current_markdown = [
            "".join(cell["source"]) for cell in current["cells"] if cell["cell_type"] == "markdown"
        ]
        assert current_markdown == original_markdown, path.name

        original_comments = [
            line.rstrip()
            for cell in original["cells"]
            if cell["cell_type"] == "code"
            for line in "".join(cell["source"]).splitlines()
            if line.lstrip().startswith("#")
        ]
        current_code = "\n".join(
            "".join(cell["source"]) for cell in current["cells"] if cell["cell_type"] == "code"
        )
        for comment in original_comments:
            assert comment in current_code, (path.name, comment)
