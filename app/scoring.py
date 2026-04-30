"""Shared readiness scoring logic.

Kept in a separate module so both :mod:`catalog_scan` (audit path) and
:mod:`query` (live API path) can import it without creating a circular dependency.

Readiness statuses (ordered by score descending):
  full_ready            100 — TradeTick rows > 0 AND OrderBookDeltas rows > 0
  l2_replay_ready        70 — OrderBookDeltas rows > 0, TradeTick absent/empty
  trade_only             60 — TradeTick rows > 0, OrderBookDeltas absent/empty
  depth10_inspection_only 50 — OrderBookDepth10 rows > 0, replay data missing
  partial_unreadable     40 — files exist but metadata/schema scan failed
  not_ready               0 — no usable rows
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .models import ScoreComponent


def readiness_status_for_presence(
    *,
    has_trade_rows: bool,
    has_delta_rows: bool,
    has_depth_rows: bool,
    partial_unreadable: bool,
) -> str:
    """Return the canonical readiness status given data-type presence flags.

    This is the authoritative function; ``readiness_status_for_score`` is kept
    only for call-sites that already have a score but not the raw flags.
    """
    if partial_unreadable:
        return "partial_unreadable"
    if has_trade_rows and has_delta_rows:
        return "full_ready"
    if has_delta_rows:
        return "l2_replay_ready"
    if has_trade_rows:
        return "trade_only"
    if has_depth_rows:
        return "depth10_inspection_only"
    return "not_ready"


def compute_readiness_breakdown(
    *,
    has_trade_tick: bool,
    has_order_book_deltas: bool,
    has_order_book_depths: bool,
    trade_row_count: int,
    delta_row_count: int,
    depth_row_count: int = 0,
    trade_max_gap_seconds: float | None = None,
    delta_max_gap_seconds: float | None = None,
    fenced_range_count: int = 0,
    desync_count: int = 0,
    resync_count: int = 0,
    session_break_count: int = 0,
    partial_unreadable: bool = False,
) -> "tuple[float, list[ScoreComponent]]":
    """Return ``(backtest_readiness_score, breakdown)``.

    Backtest readiness is intentionally only data-type completeness.
    Reliability penalties live in the L2 quality / audit-confidence scores.
    Extra parameters are accepted for backward-compatible call sites.
    """
    from .models import ScoreComponent  # local import avoids circular dep at module load

    components: list[ScoreComponent] = []

    has_trade_rows = has_trade_tick and trade_row_count > 0
    has_delta_rows = has_order_book_deltas and delta_row_count > 0
    has_depth_rows = has_order_book_depths and depth_row_count > 0

    if partial_unreadable:
        components.append(ScoreComponent(label="partial/unreadable data", points=40.0, detail="files present but scan confidence is limited", positive=True))
        return 40.0, components

    if has_trade_rows and has_delta_rows:
        components.append(ScoreComponent(label="TradeTick + OrderBookDeltas", points=100.0, detail="full Nautilus backtest data completeness", positive=True))
        if has_order_book_depths:
            components.append(ScoreComponent(label="OrderBookDepth10 present", points=0.0, detail="optional visual inspection data", positive=True))
        return 100.0, components

    if has_delta_rows:
        components.append(ScoreComponent(label="OrderBookDeltas present (no TradeTick)", points=70.0, detail="L2 replay-ready; missing TradeTick for full backtest readiness", positive=True))
        return 70.0, components

    if has_trade_rows:
        components.append(ScoreComponent(label="TradeTick present (no OrderBookDeltas)", points=60.0, detail="trade-only; missing OrderBookDeltas for full backtest readiness", positive=True))
        return 60.0, components

    if has_depth_rows:
        components.append(ScoreComponent(label="OrderBookDepth10 only", points=50.0, detail="optional visual inspection data; not usable for Nautilus backtesting", positive=True))
        return 50.0, components

    # Files present but row counts are unreadable / zero — check has_* flags
    if has_trade_tick or has_order_book_deltas or has_order_book_depths:
        # Files exist, rows are 0 or unreadable
        if partial_unreadable or (has_trade_tick and has_order_book_deltas):
            components.append(ScoreComponent(label="partial/unreadable data", points=40.0, detail="files present but rows are 0 or unreadable", positive=True))
            return 40.0, components
        if has_order_book_deltas:
            components.append(ScoreComponent(label="OrderBookDeltas files (empty/unreadable)", points=40.0, detail="delta files present with 0 rows", positive=True))
            return 40.0, components
        if has_trade_tick:
            components.append(ScoreComponent(label="TradeTick files (empty/unreadable)", points=40.0, detail="trade files present with 0 rows", positive=True))
            return 40.0, components

    return 0.0, components


def compute_readiness_score(
    *,
    has_trade_tick: bool,
    has_order_book_deltas: bool,
    has_order_book_depths: bool,
    trade_row_count: int,
    delta_row_count: int,
    depth_row_count: int = 0,
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
        depth_row_count=depth_row_count,
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
    """Map a readiness score to a status string.

    Prefer ``readiness_status_for_presence`` when the raw presence flags are
    available — this mapping cannot distinguish ``l2_replay_ready`` (score 70,
    deltas-only) from the old ``l2_ready`` label.
    """
    if score >= 100.0:
        return "full_ready"
    if score >= 70.0:
        return "l2_replay_ready"
    if score >= 60.0:
        return "trade_only"
    if score >= 50.0:
        return "depth10_inspection_only"
    if score >= 40.0:
        return "partial_unreadable"
    return "not_ready"
