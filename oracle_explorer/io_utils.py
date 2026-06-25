"""Small file IO helpers used by the oracle planner."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Iterable


def ensure_dir(path: str | Path) -> Path:
    """Create a directory and return it as a Path."""
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def read_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str | Path, data: Any, *, indent: int = 2) -> Path:
    out = Path(path)
    if out.parent:
        out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, sort_keys=True)
        f.write("\n")
    return out


def write_text_atomic(path: str | Path, text: str) -> Path:
    """Atomically write text using a sibling .tmp file and os.replace."""

    out = Path(path)
    if out.parent:
        out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(f"{out.name}.tmp")
    with tmp.open("w", encoding="utf-8") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, out)
    return out


def write_json_atomic(path: str | Path, data: Any, *, indent: int = 2) -> Path:
    text = json.dumps(data, indent=indent, sort_keys=True) + "\n"
    return write_text_atomic(path, text)


def read_jsonl(path: str | Path) -> list[Any]:
    rows: list[Any] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: str | Path, rows: Iterable[Any]) -> Path:
    out = Path(path)
    if out.parent:
        out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for row in rows:
            json.dump(row, f, sort_keys=True)
            f.write("\n")
    return out


def relative_to(path: str | Path, root: str | Path) -> str:
    path_obj = Path(path)
    root_obj = Path(root)
    try:
        return path_obj.relative_to(root_obj).as_posix()
    except ValueError:
        return path_obj.as_posix()
