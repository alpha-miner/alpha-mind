import argparse
import time
import traceback
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = ROOT / "notebooks"


def execute_notebook(path: Path) -> float:
    notebook = nbformat.read(path, as_version=4)
    client = NotebookClient(
        notebook,
        timeout=120,
        startup_timeout=60,
        kernel_name="python3",
        resources={"metadata": {"path": str(path.parent)}},
    )
    started = time.perf_counter()
    client.execute()
    return time.perf_counter() - started


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Execute Alpha-Mind notebooks")
    parser.add_argument("notebooks", nargs="*", help="Optional notebook filenames")
    args = parser.parse_args(argv)
    paths = (
        [NOTEBOOK_DIR / name for name in args.notebooks]
        if args.notebooks
        else sorted(NOTEBOOK_DIR.glob("*.ipynb"))
    )
    failures = []
    started = time.perf_counter()
    for path in paths:
        try:
            elapsed = execute_notebook(path)
        except Exception:
            failures.append(path)
            print(f"FAILED {path.name}", flush=True)
            traceback.print_exc()
        else:
            print(f"PASSED {path.name} ({elapsed:.2f}s)", flush=True)
    total = time.perf_counter() - started
    print(f"Executed {len(paths) - len(failures)}/{len(paths)} notebooks in {total:.2f}s")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
