import ast
import json
from pathlib import Path

ROOT = Path(__file__).parents[1]
NOTEBOOKS = sorted((ROOT / "notebooks").glob("*.ipynb"))
ENGLISH_CONTENT = json.loads(
    (ROOT / "tests" / "fixtures" / "notebook_english_content.json").read_text(encoding="utf-8")
)


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


def test_english_notebook_markdown_and_code_comments_are_preserved():
    for path in NOTEBOOKS:
        expected = ENGLISH_CONTENT[path.name]
        current = json.loads(path.read_text(encoding="utf-8"))
        expected_markdown = expected["markdown"]
        current_markdown = [
            "".join(cell["source"]) for cell in current["cells"] if cell["cell_type"] == "markdown"
        ]
        assert current_markdown == expected_markdown, path.name

        expected_comments = expected["code_comments"]
        current_code = "\n".join(
            "".join(cell["source"]) for cell in current["cells"] if cell["cell_type"] == "code"
        )
        for comment in expected_comments:
            assert comment in current_code, (path.name, comment)
