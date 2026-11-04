"""The container only has what the Dockerfile copies: every root module the server imports must be listed there."""

import ast
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def root_modules() -> set[str]:
    return {p.stem for p in ROOT.glob("*.py")}


def imported_root_modules(entry: str, seen: set[str] | None = None) -> set[str]:
    seen = set() if seen is None else seen
    if entry in seen:
        return seen
    seen.add(entry)
    tree = ast.parse((ROOT / f"{entry}.py").read_text())
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            names.add(node.module.split(".")[0])
    for name in names & root_modules():
        imported_root_modules(name, seen)
    return seen


def copied_by_dockerfile() -> set[str]:
    files = set()
    for line in (ROOT / "Dockerfile").read_text().splitlines():
        if line.startswith("COPY ") and ".py" in line:
            files |= {Path(tok).stem for tok in line.split()[1:-1] if tok.endswith(".py")}
    return files


def test_every_root_module_the_server_needs_is_copied_into_the_image():
    needed = imported_root_modules("server")
    missing = needed - copied_by_dockerfile()
    assert not missing, f"the Dockerfile does not copy: {sorted(missing)}"


def test_the_snapshots_and_the_store_are_in_the_image():
    text = (ROOT / "Dockerfile").read_text()
    assert "COPY RAG_data_layers/ RAG_data_layers/" in text and "COPY events_and_chunking/" in text
    assert list((ROOT / "RAG_data_layers" / "session_snapshots").glob("*.npz"))
    assert "session_snapshots" not in (ROOT / ".dockerignore").read_text()
