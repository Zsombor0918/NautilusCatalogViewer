from __future__ import annotations

import hashlib
import math
import random
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence

import numpy as np

from .models import (
    GapEntry,
    L2CheckResult,
    L2QualityResponse,
    L2Snapshot,
    L2SnapshotSummary,
    L2ViolationExample,
    OrderBookLevel,
    QualitySnapshotIssue,
)

try:
    from nautilus_trader.model.objects import FIXED_PRECISION
except Exception:  # pragma: no cover - fallback when Nautilus is unavailable.
    FIXED_PRECISION = 16


FIXED_SCALAR = float(10**FIXED_PRECISION)
SESSION_BREAK_THRESHOLD_S = 300
DEPTH_LEVELS = 10


@dataclass(slots=True)
class ParsedL2Snapshot:
    index: int
    ts_event_ns: int
    bids: list[float] = field(default_factory=list)
    asks: list[float] = field(default_factory=list)
    bid_sizes: list[float] = field(default_factory=list)
    ask_sizes: list[float] = field(default_factory=list)
    best_bid: float | None = None
    best_ask: float | None = None
    spread: float | None = None
    mid: float | None = None
    is_crossed: bool = False
    is_sorted_ok: bool = True
    has_negative_qty: bool = False
    has_zero_qty: bool = False
    has_empty_side: bool = False
    flags: int | None = None
    sequence: int | None = None
    issues: list[str] = field(default_factory=list)


def ns_to_iso(value_ns: int | None) -> str | None:
    if value_ns is None:
        return None
    seconds, nanos = divmod(int(value_ns), 1_000_000_000)
    dt = datetime.fromtimestamp(seconds, tz=timezone.utc)
    return f"{dt.strftime('%Y-%m-%dT%H:%M:%S')}.{nanos:09d}Z"


def utc_now_iso() -> str:
    return datetime.now(tz=timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def decode_fixed_decimal(value: Any) -> float:
    if value is None:
        return 0.0
    if isinstance(value, memoryview):
        raw = value.tobytes()
    elif isinstance(value, (bytes, bytearray)):
        raw = bytes(value)
    elif isinstance(value, (int, float)):
        return float(value)
    else:
        return float(value)
    return int.from_bytes(raw, byteorder="little", signed=True) / FIXED_SCALAR


def _strictly_monotonic(values: Sequence[float], *, descending: bool) -> bool:
    if len(values) < 2:
        return True
    if descending:
        return all(left > right for left, right in zip(values, values[1:]))
    return all(left < right for left, right in zip(values, values[1:]))


def _meaningful_side(levels: Sequence[tuple[float, float]]) -> list[tuple[float, float]]:
    return [(price, size) for price, size in levels if price > 0 or size > 0]


def _positive_side(levels: Sequence[tuple[float, float]]) -> list[tuple[float, float]]:
    return [(price, size) for price, size in levels if price > 0 and size >= 0]


def _evaluate_levels(
    *,
    index: int,
    ts_event_ns: int,
    bid_levels: list[tuple[float, float]],
    ask_levels: list[tuple[float, float]],
    flags: int | None = None,
    sequence: int | None = None,
) -> ParsedL2Snapshot:
    active_bids = _meaningful_side(bid_levels)
    active_asks = _meaningful_side(ask_levels)
    positive_bid_prices = [price for price, _ in active_bids if price > 0]
    positive_ask_prices = [price for price, _ in active_asks if price > 0]
    bid_sizes = [size for _, size in active_bids]
    ask_sizes = [size for _, size in active_asks]

    best_bid = positive_bid_prices[0] if positive_bid_prices else None
    best_ask = positive_ask_prices[0] if positive_ask_prices else None
    spread = (best_ask - best_bid) if best_bid is not None and best_ask is not None else None
    mid = ((best_bid + best_ask) / 2.0) if best_bid is not None and best_ask is not None else None

    is_crossed = best_bid is not None and best_ask is not None and best_bid >= best_ask
    is_sorted_ok = _strictly_monotonic(positive_bid_prices, descending=True) and _strictly_monotonic(
        positive_ask_prices,
        descending=False,
    )
    has_negative_qty = any(size < 0 for size in bid_sizes + ask_sizes)
    # Only flag zero_qty when a level has a valid price but zero size.
    # Padding entries (price=0, size=0) must not be counted.
    has_zero_qty = any(price > 0 and size == 0 for price, size in bid_levels + ask_levels)
    # Empty side requires at least one level with price > 0 AND size > 0 on each side.
    has_empty_side = (
        not any(price > 0 and size > 0 for price, size in active_bids)
        or not any(price > 0 and size > 0 for price, size in active_asks)
    )

    issues: list[str] = []
    if is_crossed:
        issues.append("crossed_book")
    if not is_sorted_ok:
        issues.append("monotonic_violation")
    if has_negative_qty:
        issues.append("negative_qty")
    if has_zero_qty:
        issues.append("zero_qty")
    if has_empty_side:
        issues.append("empty_side")

    return ParsedL2Snapshot(
        index=index,
        ts_event_ns=int(ts_event_ns),
        bids=[price for price, _ in active_bids[:DEPTH_LEVELS]],
        asks=[price for price, _ in active_asks[:DEPTH_LEVELS]],
        bid_sizes=[size for _, size in active_bids[:DEPTH_LEVELS]],
        ask_sizes=[size for _, size in active_asks[:DEPTH_LEVELS]],
        best_bid=best_bid,
        best_ask=best_ask,
        spread=spread,
        mid=mid,
        is_crossed=is_crossed,
        is_sorted_ok=is_sorted_ok,
        has_negative_qty=has_negative_qty,
        has_zero_qty=has_zero_qty,
        has_empty_side=has_empty_side,
        flags=flags,
        sequence=sequence,
        issues=issues,
    )


def parsed_snapshot_from_nautilus(snapshot: Any, index: int = 0) -> ParsedL2Snapshot:
    bid_levels = [(float(order.price), float(order.size)) for order in snapshot.bids]
    ask_levels = [(float(order.price), float(order.size)) for order in snapshot.asks]
    return _evaluate_levels(
        index=index,
        ts_event_ns=int(snapshot.ts_event),
        bid_levels=bid_levels,
        ask_levels=ask_levels,
        flags=int(getattr(snapshot, "flags", 0)),
        sequence=int(getattr(snapshot, "sequence", 0)),
    )


def parsed_snapshot_from_record(record: Mapping[str, Any], index: int = 0, depth: int = DEPTH_LEVELS) -> ParsedL2Snapshot:
    bid_levels = [
        (
            decode_fixed_decimal(record.get(f"bid_price_{level}", 0)),
            decode_fixed_decimal(record.get(f"bid_size_{level}", 0)),
        )
        for level in range(depth)
    ]
    ask_levels = [
        (
            decode_fixed_decimal(record.get(f"ask_price_{level}", 0)),
            decode_fixed_decimal(record.get(f"ask_size_{level}", 0)),
        )
        for level in range(depth)
    ]
    return _evaluate_levels(
        index=index,
        ts_event_ns=int(record.get("ts_event", 0)),
        bid_levels=bid_levels,
        ask_levels=ask_levels,
        flags=int(record["flags"]) if record.get("flags") is not None else None,
        sequence=int(record["sequence"]) if record.get("sequence") is not None else None,
    )


def parsed_to_violation_example(snapshot: ParsedL2Snapshot) -> L2ViolationExample:
    return L2ViolationExample(
        ts_event_ns=snapshot.ts_event_ns,
        ts_event_iso=ns_to_iso(snapshot.ts_event_ns) or "",
        violations=list(snapshot.issues),
        best_bid=snapshot.best_bid,
        best_ask=snapshot.best_ask,
        bids=snapshot.bids,
        asks=snapshot.asks,
        bid_sizes=snapshot.bid_sizes,
        ask_sizes=snapshot.ask_sizes,
    )


def parsed_to_snapshot(snapshot: ParsedL2Snapshot) -> L2Snapshot:
    return L2Snapshot(
        index=snapshot.index,
        ts_event_ns=snapshot.ts_event_ns,
        ts_event_iso=ns_to_iso(snapshot.ts_event_ns) or "",
        best_bid=snapshot.best_bid,
        best_ask=snapshot.best_ask,
        spread=snapshot.spread,
        mid=snapshot.mid,
        is_crossed=snapshot.is_crossed,
        is_sorted_ok=snapshot.is_sorted_ok,
        has_negative_qty=snapshot.has_negative_qty,
        has_zero_qty=snapshot.has_zero_qty,
        has_empty_side=snapshot.has_empty_side,
        issues=list(snapshot.issues),
        bids=[
            OrderBookLevel(level=index, price=price, size=size)
            for index, (price, size) in enumerate(zip(snapshot.bids, snapshot.bid_sizes))
        ],
        asks=[
            OrderBookLevel(level=index, price=price, size=size)
            for index, (price, size) in enumerate(zip(snapshot.asks, snapshot.ask_sizes))
        ],
        bid_sizes=list(snapshot.bid_sizes),
        ask_sizes=list(snapshot.ask_sizes),
        flags=snapshot.flags,
        sequence=snapshot.sequence,
    )


def parsed_to_summary(snapshot: ParsedL2Snapshot) -> L2SnapshotSummary:
    return L2SnapshotSummary(
        index=snapshot.index,
        ts_event_ns=snapshot.ts_event_ns,
        ts_event_iso=ns_to_iso(snapshot.ts_event_ns) or "",
        best_bid=snapshot.best_bid,
        best_ask=snapshot.best_ask,
        spread=snapshot.spread,
        mid=snapshot.mid,
        is_crossed=snapshot.is_crossed,
        is_sorted_ok=snapshot.is_sorted_ok,
        has_negative_qty=snapshot.has_negative_qty,
        has_zero_qty=snapshot.has_zero_qty,
        has_empty_side=snapshot.has_empty_side,
        issues=list(snapshot.issues),
    )


def parsed_to_quality_issue(snapshot: ParsedL2Snapshot) -> QualitySnapshotIssue:
    return QualitySnapshotIssue(
        index=snapshot.index,
        ts_event_ns=snapshot.ts_event_ns,
        ts_event_iso=ns_to_iso(snapshot.ts_event_ns) or "",
        issues=list(snapshot.issues),
        best_bid=snapshot.best_bid,
        best_ask=snapshot.best_ask,
        spread=snapshot.spread,
        mid=snapshot.mid,
    )


def compute_gap_entries(
    ts_values: Sequence[int],
    *,
    top_n: int = 10,
    session_break_threshold_s: int = SESSION_BREAK_THRESHOLD_S,
) -> list[GapEntry]:
    if len(ts_values) < 2:
        return []

    threshold_ns = session_break_threshold_s * 1_000_000_000
    diffs = np.diff(np.asarray(ts_values, dtype=np.int64))
    order = np.argsort(diffs)[::-1][:top_n]
    entries: list[GapEntry] = []
    for diff_index in order:
        gap_ns = int(diffs[diff_index])
        prev_ts = int(ts_values[diff_index])
        next_ts = int(ts_values[diff_index + 1])
        entries.append(
            GapEntry(
                gap_ns=gap_ns,
                gap_seconds=gap_ns / 1_000_000_000,
                previous_ts_event_ns=prev_ts,
                previous_ts_event_iso=ns_to_iso(prev_ts) or "",
                next_ts_event_ns=next_ts,
                next_ts_event_iso=ns_to_iso(next_ts) or "",
                source_file=None,
                is_session_break=gap_ns >= threshold_ns,
            ),
        )
    return entries


def estimate_missing_ratio(ts_values: Sequence[int]) -> float:
    if len(ts_values) < 3:
        return 0.0
    diffs = np.diff(np.asarray(ts_values, dtype=np.int64))
    positive = diffs[diffs > 0]
    if positive.size == 0:
        return 0.0
    duration_ns = int(ts_values[-1] - ts_values[0])
    if duration_ns <= 0:
        return 0.0
    baseline = int(np.median(positive))
    outlier_floor = max(baseline * 5, 1)
    missing_ns = int(np.sum(np.maximum(0, positive[positive > outlier_floor] - baseline)))
    return round(min(1.0, missing_ns / duration_ns), 6)


def compute_l2_quality_score(
    *,
    snapshot_count: int,
    crossed_count: int,
    monotonic_violation_count: int,
    negative_qty_count: int,
    zero_qty_count: int,
    empty_side_count: int,
    max_gap_seconds: float | None,
    session_break_count: int,
    desync_count: int = 0,
    fenced_range_count: int = 0,
    resync_count: int = 0,
    bad_lines: int = 0,
    missing_symbol_count: int = 0,
    partial_unreadable_count: int = 0,
) -> float:
    """Return an L2/data reliability score.

    The base penalty model is the original depth10 quality model:
    crossed books, monotonic violations, bad quantities, empty sides,
    max gaps and session breaks. Optional converter-report signals extend
    the same reliability score when available.
    """
    if snapshot_count <= 0:
        base_score = 100.0
    else:
        crossed_rate = crossed_count / snapshot_count
        monotonic_rate = monotonic_violation_count / snapshot_count
        negative_rate = negative_qty_count / snapshot_count
        zero_rate = zero_qty_count / snapshot_count
        empty_rate = empty_side_count / snapshot_count

        penalty = (
            crossed_rate * 70.0
            + monotonic_rate * 35.0
            + negative_rate * 25.0
            + zero_rate * 8.0
            + empty_rate * 20.0
        )
        if max_gap_seconds is not None and max_gap_seconds > 0:
            penalty += min(20.0, math.log10(max_gap_seconds + 1.0) * 4.0)
        penalty += min(15.0, session_break_count * 3.0)
        base_score = 100.0 - penalty

    converter_penalty = 0.0
    converter_penalty += min(10.0, desync_count * 2.0)
    converter_penalty += min(15.0, fenced_range_count * 1.5)
    converter_penalty += min(5.0, resync_count * 0.5)
    if bad_lines > 0:
        converter_penalty += min(10.0, math.log10(bad_lines + 1.0) * 3.0)
    converter_penalty += min(10.0, missing_symbol_count * 2.0)
    converter_penalty += min(20.0, partial_unreadable_count * 5.0)
    return round(max(0.0, base_score - converter_penalty), 2)


def compute_quality_score(**kwargs) -> float:
    """Backward-compatible alias for the L2/data reliability score."""
    return compute_l2_quality_score(**kwargs)


def quality_from_snapshots(
    *,
    instrument_id: str,
    instrument_type: str,
    snapshots: Sequence[ParsedL2Snapshot],
    from_ns: int | None = None,
    to_ns: int | None = None,
    top_n: int = 10,
    example_limit: int = 20,
    session_break_threshold_s: int = SESSION_BREAK_THRESHOLD_S,
) -> L2QualityResponse:
    ts_values = [snapshot.ts_event_ns for snapshot in snapshots]
    snapshot_count = len(snapshots)
    crossed_count = sum(snapshot.is_crossed for snapshot in snapshots)
    monotonic_violation_count = sum(not snapshot.is_sorted_ok for snapshot in snapshots)
    negative_qty_count = sum(snapshot.has_negative_qty for snapshot in snapshots)
    zero_qty_count = sum(snapshot.has_zero_qty for snapshot in snapshots)
    empty_side_count = sum(snapshot.has_empty_side for snapshot in snapshots)
    bad_snapshots = [snapshot for snapshot in snapshots if snapshot.issues]
    bad_snapshot_count = len(bad_snapshots)
    gap_entries = compute_gap_entries(
        ts_values,
        top_n=top_n,
        session_break_threshold_s=session_break_threshold_s,
    )
    max_gap_seconds = gap_entries[0].gap_seconds if gap_entries else None
    threshold_ns = session_break_threshold_s * 1_000_000_000
    session_break_count = 0
    if len(ts_values) > 1:
        diffs = np.diff(np.asarray(ts_values, dtype=np.int64))
        session_break_count = int(np.sum(diffs >= threshold_ns))
    quality_score = compute_l2_quality_score(
        snapshot_count=snapshot_count,
        crossed_count=crossed_count,
        monotonic_violation_count=monotonic_violation_count,
        negative_qty_count=negative_qty_count,
        zero_qty_count=zero_qty_count,
        empty_side_count=empty_side_count,
        max_gap_seconds=max_gap_seconds,
        session_break_count=session_break_count,
    )

    return L2QualityResponse(
        instrument_id=instrument_id,
        instrument_type=instrument_type,
        from_ns=from_ns if from_ns is not None else (ts_values[0] if ts_values else None),
        from_iso=ns_to_iso(from_ns if from_ns is not None else (ts_values[0] if ts_values else None)),
        to_ns=to_ns if to_ns is not None else (ts_values[-1] if ts_values else None),
        to_iso=ns_to_iso(to_ns if to_ns is not None else (ts_values[-1] if ts_values else None)),
        generated_at=utc_now_iso(),
        snapshot_count=snapshot_count,
        crossed_count=int(crossed_count),
        crossed_rate=(crossed_count / snapshot_count) if snapshot_count else 0.0,
        monotonic_violation_count=int(monotonic_violation_count),
        monotonic_violation_rate=(monotonic_violation_count / snapshot_count) if snapshot_count else 0.0,
        negative_qty_count=int(negative_qty_count),
        zero_qty_count=int(zero_qty_count),
        empty_side_count=int(empty_side_count),
        empty_side_rate=(empty_side_count / snapshot_count) if snapshot_count else 0.0,
        bad_snapshot_count=bad_snapshot_count,
        bad_snapshot_rate=(bad_snapshot_count / snapshot_count) if snapshot_count else 0.0,
        l2_quality_score=quality_score,
        data_reliability_score=quality_score,
        quality_score=quality_score,
        max_gap_seconds=max_gap_seconds,
        session_break_count=session_break_count,
        session_break_threshold_s=session_break_threshold_s,
        top_gaps=gap_entries,
        bad_snapshots=[parsed_to_quality_issue(snapshot) for snapshot in bad_snapshots[:example_limit]],
    )


def _sample_indices(total_count: int, first_n: int, random_n: int, instrument_id: str) -> list[int]:
    first_indices = list(range(min(first_n, total_count)))
    remaining_indices = list(range(len(first_indices), total_count))
    seed = int(hashlib.sha256(instrument_id.encode("utf-8")).hexdigest()[:16], 16)
    rng = random.Random(seed)
    random_indices = rng.sample(remaining_indices, k=min(random_n, len(remaining_indices))) if remaining_indices else []
    return sorted(set(first_indices + random_indices))


def run_l2_checks(
    *,
    instrument_id: str,
    snapshots: Sequence[Any],
    instrument_type: str = "unknown",
    first_n: int = 10,
    random_n: int = 10,
    example_limit: int = 5,
) -> L2CheckResult:
    snapshot_count = len(snapshots)
    if snapshot_count == 0:
        return L2CheckResult(present=False, snapshot_count=0, sampled_count=0)

    indices = _sample_indices(snapshot_count, first_n, random_n, instrument_id)
    parsed_snapshots = [
        snapshot if isinstance(snapshot, ParsedL2Snapshot) else parsed_snapshot_from_nautilus(snapshot, index=index)
        for index, snapshot in ((index, snapshots[index]) for index in indices)
    ]
    quality = quality_from_snapshots(
        instrument_id=instrument_id,
        instrument_type=instrument_type,
        snapshots=parsed_snapshots,
        top_n=10,
        example_limit=example_limit,
    )

    return L2CheckResult(
        present=True,
        snapshot_count=snapshot_count,
        sampled_count=len(parsed_snapshots),
        bad_count=quality.bad_snapshot_count,
        bad_rate=quality.bad_snapshot_rate,
        crossed_count=quality.crossed_count,
        monotonic_violation_count=quality.monotonic_violation_count,
        negative_qty_count=quality.negative_qty_count,
        zero_qty_count=quality.zero_qty_count,
        empty_side_count=quality.empty_side_count,
        l2_quality_score=quality.l2_quality_score,
        data_reliability_score=quality.data_reliability_score,
        quality_score=quality.quality_score,
        top_gaps=quality.top_gaps,
        examples=[parsed_to_violation_example(snapshot) for snapshot in parsed_snapshots if snapshot.issues][:example_limit],
    )
