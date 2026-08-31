from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from moprm.schema import ProblemRecord


def load_jsonl(path: str | Path) -> list[ProblemRecord]:
    records: list[ProblemRecord] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                data = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}") from exc
            records.append(ProblemRecord.from_dict(data))
    return records


def write_jsonl(path: str | Path, records: Iterable[dict]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False))
            handle.write("\n")

