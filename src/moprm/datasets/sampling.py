from __future__ import annotations

import random
from collections import defaultdict

from moprm.schema import ProblemRecord


SourceQuota = tuple[str, str, int]


def parse_source_quota(text: str) -> SourceQuota:
    """Parse a quota in the form ``domain|metadata.source=count``."""

    if "=" not in text:
        raise ValueError(
            f"Invalid source quota {text!r}; expected domain|source=count"
        )
    lhs, count_text = text.rsplit("=", 1)
    if "|" not in lhs:
        raise ValueError(
            f"Invalid source quota {text!r}; expected domain|source=count"
        )
    domain, source = lhs.split("|", 1)
    domain = domain.strip()
    source = source.strip()
    if not domain or not source:
        raise ValueError(
            f"Invalid source quota {text!r}; domain and source must be non-empty"
        )
    try:
        count = int(count_text)
    except ValueError as exc:
        raise ValueError(f"Invalid source quota count in {text!r}") from exc
    if count < 0:
        raise ValueError(f"Invalid source quota count in {text!r}; count must be >= 0")
    return domain, source, count


def sample_records_by_source_quotas(
    records: list[ProblemRecord],
    quotas: list[SourceQuota],
    seed: int = 13,
    exclude_ids: set[str] | None = None,
) -> list[ProblemRecord]:
    rng = random.Random(seed)
    excluded = exclude_ids or set()
    by_key: dict[tuple[str, str], list[ProblemRecord]] = defaultdict(list)
    for record in records:
        if record.problem_id in excluded:
            continue
        source = str(record.metadata.get("source", ""))
        by_key[(record.domain, source)].append(record)

    sampled: list[ProblemRecord] = []
    used_ids: set[str] = set()
    for domain, source, count in quotas:
        pool = list(by_key.get((domain, source), []))
        rng.shuffle(pool)
        selected = [record for record in pool if record.problem_id not in used_ids][:count]
        if len(selected) < count:
            raise ValueError(
                "Not enough records for source quota "
                f"{domain}|{source}={count}; found {len(pool)}"
            )
        sampled.extend(selected)
        used_ids.update(record.problem_id for record in selected)

    rng.shuffle(sampled)
    return sampled


def sample_records(
    records: list[ProblemRecord],
    limit: int | None = None,
    per_domain: int | None = None,
    seed: int = 13,
    exclude_ids: set[str] | None = None,
) -> list[ProblemRecord]:
    excluded = exclude_ids or set()
    records = [record for record in records if record.problem_id not in excluded]

    if limit is None and per_domain is None:
        return list(records)

    rng = random.Random(seed)
    if per_domain is not None:
        by_domain: dict[str, list[ProblemRecord]] = defaultdict(list)
        for record in records:
            by_domain[record.domain].append(record)
        sampled: list[ProblemRecord] = []
        for domain in sorted(by_domain):
            domain_records = list(by_domain[domain])
            rng.shuffle(domain_records)
            sampled.extend(domain_records[:per_domain])
        rng.shuffle(sampled)
    else:
        sampled = list(records)
        rng.shuffle(sampled)

    if limit is not None:
        sampled = sampled[:limit]
    return sampled
