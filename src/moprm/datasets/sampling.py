from __future__ import annotations

import random
from collections import defaultdict

from moprm.schema import ProblemRecord


def sample_records(
    records: list[ProblemRecord],
    limit: int | None = None,
    per_domain: int | None = None,
    seed: int = 13,
) -> list[ProblemRecord]:
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

