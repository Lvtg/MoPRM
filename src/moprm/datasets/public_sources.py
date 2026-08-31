from __future__ import annotations

import json
import random
import re
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from moprm.io import write_jsonl


USER_AGENT = "MoPRM/0.1 dataset-preparation"
HF_ROWS_ENDPOINT = "https://datasets-server.huggingface.co/rows"
BBH_RAW_BASE = "https://raw.githubusercontent.com/suzgunmirac/BIG-Bench-Hard/main/bbh"


@dataclass(frozen=True)
class PublicSubsetConfig:
    math500_limit: int = 80
    gsm8k_limit: int = 80
    bbh_limit_per_task: int = 60
    seed: int = 13


def fetch_json(url: str, timeout: int = 90) -> dict[str, Any]:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def fetch_hf_rows(
    dataset: str,
    config: str,
    split: str,
    limit: int,
    offset: int = 0,
    page_size: int = 100,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    while len(rows) < limit:
        length = min(page_size, limit - len(rows))
        params = urllib.parse.urlencode(
            {
                "dataset": dataset,
                "config": config,
                "split": split,
                "offset": offset + len(rows),
                "length": length,
            }
        )
        payload = fetch_json(f"{HF_ROWS_ENDPOINT}?{params}")
        page_rows = payload.get("rows", [])
        if not page_rows:
            break
        rows.extend(item["row"] for item in page_rows)
    return rows[:limit]


def extract_gsm8k_answer(answer: str) -> str:
    marker = "####"
    if marker in answer:
        return answer.split(marker)[-1].strip()
    numbers = re.findall(r"[-+]?\d+(?:,\d{3})*(?:\.\d+)?", answer)
    return numbers[-1].replace(",", "") if numbers else answer.strip()


def normalize_choice(answer: str) -> str:
    matches = re.findall(r"(?<![A-Z])([A-E])(?![A-Z])", str(answer), flags=re.IGNORECASE)
    return matches[-1].upper() if matches else str(answer).strip()


def make_problem(
    problem_id: str,
    domain: str,
    problem: str,
    answer: str,
    source: str,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    merged_metadata = {"source": source}
    if metadata:
        merged_metadata.update(metadata)
    return {
        "problem_id": problem_id,
        "domain": domain,
        "problem": problem,
        "answer": answer,
        "candidates": [],
        "metadata": merged_metadata,
    }


def prepare_math500(limit: int) -> list[dict[str, Any]]:
    rows = fetch_hf_rows("HuggingFaceH4/MATH-500", "default", "test", limit)
    records: list[dict[str, Any]] = []
    for idx, row in enumerate(rows):
        answer = str(row.get("answer") or row.get("solution") or "").strip()
        records.append(
            make_problem(
                problem_id=f"math500_{idx:04d}",
                domain="math",
                problem=str(row.get("problem", "")).strip(),
                answer=answer,
                source="HuggingFaceH4/MATH-500",
                metadata={
                    "subject": row.get("subject"),
                    "level": row.get("level"),
                    "unique_id": row.get("unique_id"),
                },
            )
        )
    return records


def prepare_gsm8k(limit: int) -> list[dict[str, Any]]:
    rows = fetch_hf_rows("openai/gsm8k", "main", "test", limit)
    records: list[dict[str, Any]] = []
    for idx, row in enumerate(rows):
        full_answer = str(row.get("answer", "")).strip()
        records.append(
            make_problem(
                problem_id=f"gsm8k_{idx:04d}",
                domain="math",
                problem=str(row.get("question", "")).strip(),
                answer=extract_gsm8k_answer(full_answer),
                source="openai/gsm8k",
                metadata={"full_solution": full_answer},
            )
        )
    return records


def prepare_bbh_logical_deduction(limit_per_task: int, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    task_names = [
        "logical_deduction_three_objects",
        "logical_deduction_five_objects",
        "logical_deduction_seven_objects",
    ]
    records: list[dict[str, Any]] = []
    for task_name in task_names:
        payload = fetch_json(f"{BBH_RAW_BASE}/{task_name}.json")
        examples = list(payload.get("examples", []))
        rng.shuffle(examples)
        for idx, row in enumerate(examples[:limit_per_task]):
            records.append(
                make_problem(
                    problem_id=f"bbh_{task_name}_{idx:04d}",
                    domain="logic",
                    problem=str(row.get("input", "")).strip(),
                    answer=normalize_choice(str(row.get("target", ""))),
                    source=f"BIG-Bench-Hard/{task_name}",
                    metadata={"task_name": task_name},
                )
            )
    return records


def prepare_public_subsets(output_dir: str | Path, config: PublicSubsetConfig) -> dict[str, Path]:
    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    math500 = prepare_math500(config.math500_limit)
    gsm8k = prepare_gsm8k(config.gsm8k_limit)
    bbh_logic = prepare_bbh_logical_deduction(config.bbh_limit_per_task, config.seed)
    combined = math500 + gsm8k + bbh_logic

    outputs = {
        "math500": target_dir / "math500.jsonl",
        "gsm8k": target_dir / "gsm8k.jsonl",
        "bbh_logic": target_dir / "bbh_logic.jsonl",
        "combined": target_dir / "math_logic_combined.jsonl",
    }
    write_jsonl(outputs["math500"], math500)
    write_jsonl(outputs["gsm8k"], gsm8k)
    write_jsonl(outputs["bbh_logic"], bbh_logic)
    write_jsonl(outputs["combined"], combined)
    return outputs
