from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable


@dataclass(frozen=True)
class RankedItem:
    id: str
    payload: dict[str, Any]
    score: float


def reciprocal_rank_fusion(
    *rankings: Iterable[RankedItem],
    k: int = 60,
    top_n: int = 5,
) -> list[RankedItem]:
    """
    Combine multiple ranked lists using Reciprocal Rank Fusion (RRF).

    Score(item) = sum(1 / (k + rank_i(item))) across rankings where it appears.
    """
    fused: dict[str, float] = {}
    payloads: dict[str, dict[str, Any]] = {}

    for ranking in rankings:
        for rank, item in enumerate(ranking, start=1):
            fused[item.id] = fused.get(item.id, 0.0) + 1.0 / (k + rank)
            payloads.setdefault(item.id, item.payload)

    out = [
        RankedItem(id=_id, payload=payloads[_id], score=score)
        for _id, score in fused.items()
    ]
    out.sort(key=lambda x: x.score, reverse=True)
    return out[:top_n]

