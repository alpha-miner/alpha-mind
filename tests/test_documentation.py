import ast
import re
from pathlib import Path

ROOT = Path(__file__).parents[1]
DOCUMENTS = [
    ROOT / "README.md",
    *sorted((ROOT / "docs").glob("*.md")),
    *sorted((ROOT / "examples" / "parquet_demo").glob("*.md")),
]


def test_markdown_links_and_code_fences():
    errors = []
    for path in DOCUMENTS:
        text = path.read_text(encoding="utf-8")
        if text.count("```") % 2:
            errors.append(f"{path}: unbalanced code fences")
        for match in re.finditer(r"\[[^]]+\]\(([^)]+)\)", text):
            target = match.group(1).split("#", 1)[0]
            if not target or target.startswith(("http://", "https://", "mailto:")):
                continue
            if not (path.parent / target).resolve().exists():
                errors.append(f"{path}: broken link {target}")
    assert not errors, "\n".join(errors)


def test_markdown_python_blocks_have_valid_syntax():
    errors = []
    for path in DOCUMENTS:
        text = path.read_text(encoding="utf-8")
        for index, match in enumerate(re.finditer(r"```python\n(.*?)```", text, re.S), 1):
            try:
                ast.parse(match.group(1))
            except SyntaxError as error:
                errors.append(f"{path}: Python block {index}: {error}")
    assert not errors, "\n".join(errors)


def test_document_index_covers_every_feature_page():
    index = (ROOT / "docs" / "README.md").read_text(encoding="utf-8")
    for path in (ROOT / "docs").glob("*.md"):
        if path.name != "README.md":
            assert f"({path.name})" in index
