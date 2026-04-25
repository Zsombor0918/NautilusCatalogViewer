"""Shared readiness scoring logic.

Kept in a separate module so both :mod:`catalog_scan` (audit path) and
:mod:`query` (live API path) can import it without creating a circular dependency.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .models import ScoreComponent


def compute_readiness_breakdown(
    *,
    has_trade_tick: bool,
    has_order_book_deltas: bool,
    has_order_book_depths: bool,
    trade_row_count: int,
    delta_row_count: int,
    trade_max_gap_seconds: float | None,
    delta_max_gap_seconds: float | None,
    fenced_range_count: int,
    desync_count: int,
    resync_count: int,
    session_break_count: int,
) -> "tuple[float, list[ScoreComponent]]":
    """Return ``(final_score, breakdown)`` — every scored bonus and penalty.

    Scoring table
    ─────────────
    trade_tick present          +20
    order_book_deltas present   +25
    order_book_depths present    +5
    trade rows > 1 000          +10  (> 0 but ≤ 1 000: +5)
    delta rows > 1 000          +10  (> 0 but ≤ 1 000: +5)

    Penalties (all capped)
    ──────────────────────
    trade gap (log-scaled)      up to −7.5
    delta gap (log-scaled)      up to −7.5
    fenced ranges               −2 each, max −10
    desync events               −1 each, max −5
    resync events               −0.5 each, max −5
    session breaks              −0.5 each, max −5
    """
    from .models import ScoreComponent  # local import avoids circular dep at module load

    components: list[ScoreComponent] = []

    if has_trade_tick:
        components.append(ScoreComponent(label="trade_tick present", points=20.0, detail="", positive=True))
    if has_order_book_deltas:
        components.append(ScoreComponent(label="order_book_deltas present", points=25.0, detail="", positive=True))
    if has_order_book_depths:
        components.append(ScoreComponent(label="order_book_depths present", points=5.0, detail="optional bonus", positive=True))

    if trade_row_count > 1000:
        components.append(ScoreComponent(label="trade rows \u2265 1\u202f000", points=10.0, detail=f"{trade_row_count:,} rows", positive=True))
    elif trade_row_count > 0:
        components.append(ScoreComponent(label="trade rows > 0", points=5.0, detail=f"{trade_row_count:,} rows", positive=True))

    if delta_row_count > 1000:
        components.append(ScoreComponent(label="delta rows \u2265 1\u202f000", points=10.0, detail=f"{delta_row_count:,} rows", positive=True))
    elif delta_row_count > 0:
        components.append(ScoreComponent(label="delta rows > 0", points=5.0, detail=f"{delta_row_count:,} rows", positive=True))

    for label_prefix, gap_sec in (("trade", trade_max_gap_seconds), ("delta", delta_max_gap_seconds)):
        if gap_sec is not None and gap_sec > 0:
            penalty = round(min(7.5, math.log10(gap_sec + 1.0) * 2.5), 2)
            components.append(ScoreComponent(
                label=f"{label_prefix} gap penalty",
                points=-penalty,
                detail=f"max gap {gap_sec:,.1f} s",
                positive=False,
            ))

    if fenced_range_count > 0:
        penalty = round(min(10.0, fenced_range_count * 2.0), 2)
        components.append(ScoreComponent(
            label="fenced ranges",
            points=-penalty,
            detail=f"{fenced_range_count} range(s), \u22122 each (cap \u221210)",
            positive=False,
        ))

    if desync_count > 0:
        penalty = round(min(5.0, desync_count * 1.0), 2)
        components.append(ScoreComponent(
            label="desync events",
            points=-penalty,
            detail=f"{desync_count} event(s), \u22121 each (cap \u22125)",
            positive=False,
        ))

    if resync_count > 0:
        penalty = round(min(5.0, resync_count * 0.5), 2)
        components.append(ScoreComponent(
            label="resync events",
            points=-penalty,
            detail=f"{resync_count} event(s), \u22120.5 each (cap \u22125)",
            positive=False,
        ))

    if session_break_count > 0:
        penalty = round(min(5.0, session_break_count * 0.5), 2)
        components.append(ScoreComponent(
            label="session breaks",
            points=-penalty,
            detail=f"{session_break_count} break(s), \u22120.5 each (cap \u22125)",
            positive=False,
        ))

    raw = sum(c.points for c in components)
    final = round(max(0.0, min(100.0, raw)), 2)
    return final, components


def compute_readiness_score(
    *,
    has_trade_tick: bool,
    has_order_book_deltas: bool,
    has_order_book_depths: bool,
    trade_row_count: int,
    delta_row_count: int,
    trade_max_gap_seconds: float | None,
    delta_max_gap_seconds: float | None,
    fenced_range_count: int,
    desync_count: int,
    resync_count: int,
    session_break_count: int,
) -> float:
    """Convenience wrapper — returns only the final score."""
    score, _ = compute_readiness_breakdown(
        has_trade_tick=has_trade_tick,
        has_order_book_deltas=has_order_book_deltas,
        has_order_book_depths=has_order_book_depths,
        trade_row_count=trade_row_count,
        delta_row_count=delta_row_count,
        trade_max_gap_seconds=trade_max_gap_seconds,
        delta_max_gap_seconds=delta_max_gap_seconds,
        fenced_range_count=fenced_range_count,
        desync_count=desync_count,
        resync_count=resync_count,
        session_break_count=session_break_count,
    )
    return score
