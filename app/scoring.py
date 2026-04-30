"""Shared readiness scoring logic.

Kept in a separate module so both :mod:`catalog_scan` (audit path) and
:mod:`query` (live API path) can import it without creating a circular dependency.
"""

from __future__ import annotations

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
    trade_max_gap_seconds: float | None = None,
    delta_max_gap_seconds: float | None = None,
    fenced_range_count: int = 0,
    desync_count: int = 0,
    resync_count: int = 0,
    session_break_count: int = 0,
    partial_unreadable: bool = False,
) -> "tuple[float, list[ScoreComponent]]":
    """Return ``(backtest_readiness_score, breakdown)``.

    Backtest readiness is intentionally only data-type completeness:
    full_ready=100, l2_ready=70, trade_ready=60, partial_unreadable=40,
    not_ready=0. Reliability penalties live in the L2 quality score.
    Extra parameters are accepted for backward-compatible call sites.
    """
    from .models import ScoreComponent  # local import avoids circular dep at module load

    components: list[ScoreComponent] = []

    if partial_unreadable:
        components.append(ScoreComponent(label="partial/unreadable data", points=40.0, detail="files present but scan confidence is limited", positive=True))
        return 40.0, components

    if has_trade_tick and has_order_book_deltas:
        components.append(ScoreComponent(label="TradeTick + OrderBookDeltas", points=100.0, detail="full Nautilus backtest data completeness", positive=True))
        if has_order_book_depths:
            components.append(ScoreComponent(label="OrderBookDepth10 present", points=0.0, detail="optional inspection data", positive=True))
        return 100.0, components

    if has_order_book_depths:
        components.append(ScoreComponent(label="OrderBookDepth10 present", points=70.0, detail="L2 snapshots available without TradeTick + OrderBookDeltas completeness", positive=True))
        return 70.0, components

    if has_order_book_deltas:
        components.append(ScoreComponent(label="OrderBookDeltas present", points=70.0, detail="L2-ready, missing TradeTick for full backtest readiness", positive=True))
        return 70.0, components

    if has_trade_tick:
        components.append(ScoreComponent(label="TradeTick present", points=60.0, detail="trade-ready, missing L2 data for full backtest readiness", positive=True))
        return 60.0, components

    return 0.0, components


def compute_readiness_score(
    *,
    has_trade_tick: bool,
    has_order_book_deltas: bool,
    has_order_book_depths: bool,
    trade_row_count: int,
    delta_row_count: int,
    trade_max_gap_seconds: float | None = None,
    delta_max_gap_seconds: float | None = None,
    fenced_range_count: int = 0,
    desync_count: int = 0,
    resync_count: int = 0,
    session_break_count: int = 0,
    partial_unreadable: bool = False,
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
        partial_unreadable=partial_unreadable,
    )
    return score


def readiness_status_for_score(score: float) -> str:
    if score >= 100.0:
        return "full_ready"
    if score >= 70.0:
        return "l2_ready"
    if score >= 60.0:
        return "trade_ready"
    if score >= 40.0:
        return "partial_unreadable"
    return "not_ready"
