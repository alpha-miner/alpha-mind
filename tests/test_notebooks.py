import ast
import json
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
        text = path.read_text(encoding="utf-8")
        for legacy in ("PyFin", "pandas", "cvxopt", "DB_URI"):
            assert legacy not in text, (path.name, legacy)


def test_notebook_index_covers_every_notebook():
    index = (ROOT / "notebooks" / "README.md").read_text(encoding="utf-8")
    for path in NOTEBOOKS:
        assert f"<{path.name}>" in index
