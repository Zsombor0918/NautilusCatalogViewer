from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


# ── Shared ──────────────────────────────────────────────────────────────────

class CatalogWarning(StrictModel):
    code: str
    message: str
    path: str | None = None


class GapEntry(StrictModel):
    gap_ns: int
    gap_seconds: float
    previous_ts_event_ns: int
    previous_ts_event_iso: str
    next_ts_event_ns: int
    next_ts_event_iso: str
    source_file: str | None = None
    is_session_break: bool = False


# ── Report models (deterministic recorder/converter reports) ────────────────

class FencedRange(StrictModel):
    """A fenced interval where data may be incomplete or unreliable."""
    start_ns: int
    start_iso: str
    end_ns: int
    end_iso: str
    reason: str = ""


class SessionBoundary(StrictModel):
    """Session start/end marker from the recorder report."""
    ts_ns: int
    ts_iso: str
    kind: Literal["start", "end"] = "start"
    label: str = ""


class ResyncEvent(StrictModel):
    """A resync/snapshot-seed/desync event from the converter report."""
    ts_ns: int
    ts_iso: str
    kind: Literal["snapshot_seed", "resync", "desync"] = "resync"
    detail: str = ""


class ReportContext(StrictModel):
    """Deterministic recorder/converter report summary for one instrument."""
    report_found: bool = False
    snapshot_seed_count: int = 0
    resync_count: int = 0
    desync_count: int = 0
    fenced_ranges: list[FencedRange] = Field(default_factory=list)
    session_boundaries: list[SessionBoundary] = Field(default_factory=list)
    resync_events: list[ResyncEvent] = Field(default_factory=list)
    last_committed_update_id: str | None = None
    trade_id_diagnostics: list[str] = Field(default_factory=list)
    converter_warnings: list[str] = Field(default_factory=list)


# ── Inventory models ────────────────────────────────────────────────────────

class InstrumentTypeSummary(StrictModel):
    instrument_type: str
    instrument_count: int
    with_any_data_count: int = 0
    instruments: list[str] = Field(default_factory=list)


class InstrumentCoverage(StrictModel):
    data_type: str
    present: bool = False
    file_count: int = 0
    row_count_estimate: int = 0
    ts_event_min_ns: int | None = None
    ts_event_min_iso: str | None = None
    ts_event_max_ns: int | None = None
    ts_event_max_iso: str | None = None
    max_gap_seconds: float | None = None
    missing_ratio_estimate: float = 0.0
    session_break_count: int = 0
    corrupt: bool = False
    error: str | None = None


class InstrumentInventoryItem(StrictModel):
    instrument_id: str
    instrument_type: str
    has_any_data: bool = False
    coverage: dict[str, InstrumentCoverage] = Field(default_factory=dict)


class InventoryResponse(StrictModel):
    catalog_root: str
    generated_at: str
    available_data_types: list[str] = Field(default_factory=list)
    instrument_types: list[InstrumentTypeSummary] = Field(default_factory=list)
    instruments: list[InstrumentInventoryItem] = Field(default_factory=list)
    warnings: list[CatalogWarning] = Field(default_factory=list)


# ── Readiness model (deterministic-first) ──────────────────────────────────

class ReadinessResult(StrictModel):
    """Per-instrument deterministic readiness assessment."""
    instrument_id: str = ""
    instrument_type: str = ""
    # Primary data availability
    has_trade_tick: bool = False
    has_order_book_deltas: bool = False
    has_order_book_depths: bool = False  # optional / secondary
    delta_first_only: bool = False
    # Readiness signals
    is_consumable: bool = False
    is_backtest_ready: bool = False
    # Coverage
    trade_row_count: int = 0
    delta_row_count: int = 0
    depth_row_count: int = 0
    trade_duration_seconds: float | None = None
    delta_duration_seconds: float | None = None
    # Integrity signals
    trade_max_gap_seconds: float | None = None
    delta_max_gap_seconds: float | None = None
    session_break_count: int = 0
    # Report context
    fenced_range_count: int = 0
    resync_count: int = 0
    desync_count: int = 0
    snapshot_seed_count: int = 0
    # Readiness score (0-100, deterministic-first)
    readiness_score: float = 0.0
    limitations: list[str] = Field(default_factory=list)
    report: ReportContext = Field(default_factory=ReportContext)


class ReadinessResponse(StrictModel):
    instrument_id: str
    instrument_type: str
    generated_at: str
    readiness: ReadinessResult
    warnings: list[CatalogWarning] = Field(default_factory=list)


# ── Audit models (deterministic-first) ──────────────────────────────────────

class DataTypeAuditStats(StrictModel):
    data_type: str
    present: bool = False
    file_count: int = 0
    row_count_estimate: int = 0
    ts_event_min_ns: int | None = None
    ts_event_min_iso: str | None = None
    ts_event_max_ns: int | None = None
    ts_event_max_iso: str | None = None
    duration_seconds: float | None = None
    max_gap_ns: int | None = None
    max_gap_seconds: float | None = None
    missing_ratio_estimate: float = 0.0
    session_break_count: int = 0
    top_gaps: list[GapEntry] = Field(default_factory=list)
    corrupt: bool = False
    error: str | None = None


# L2 check models (secondary — only relevant when depth10 is present)
class L2ViolationExample(StrictModel):
    ts_event_ns: int
    ts_event_iso: str
    violations: list[str] = Field(default_factory=list)
    best_bid: float | None = None
    best_ask: float | None = None
    bids: list[float] = Field(default_factory=list)
    asks: list[float] = Field(default_factory=list)
    bid_sizes: list[float] = Field(default_factory=list)
    ask_sizes: list[float] = Field(default_factory=list)


class L2CheckResult(StrictModel):
    """Optional L2 depth10 structural check (secondary)."""
    present: bool = False
    snapshot_count: int = 0
    sampled_count: int = 0
    bad_count: int = 0
    bad_rate: float = 0.0
    crossed_count: int = 0
    monotonic_violation_count: int = 0
    negative_qty_count: int = 0
    zero_qty_count: int = 0
    empty_side_count: int = 0
    quality_score: float = 100.0
    top_gaps: list[GapEntry] = Field(default_factory=list)
    examples: list[L2ViolationExample] = Field(default_factory=list)
    error: str | None = None


class AuditInstrumentResult(StrictModel):
    instrument_id: str
    instrument_type: str
    data_types: dict[str, DataTypeAuditStats] = Field(default_factory=dict)
    # Deterministic-first readiness (primary)
    readiness: ReadinessResult = Field(default_factory=ReadinessResult)
    # Optional L2 check (secondary — only when depth10 is present)
    l2_check: L2CheckResult = Field(default_factory=L2CheckResult)
    # Legacy quality fields (demoted, derived from L2 when present)
    quality_score: float = 100.0
    quality_snapshot_count: int = 0
    quality_bad_snapshot_count: int = 0
    crossed_count: int = 0
    monotonic_violation_count: int = 0
    empty_side_count: int = 0
    crossed_rate: float = 0.0
    empty_rate: float = 0.0
    session_break_count: int = 0
    max_gap_seconds: float | None = None
    corrupt: bool = False


class ChartPoint(StrictModel):
    instrument_id: str
    instrument_type: str
    data_type: str
    max_gap_seconds: float


class ReadinessOffenderItem(StrictModel):
    """Readiness-context offender item (primary)."""
    instrument_id: str
    instrument_type: str
    readiness_score: float = 0.0
    has_trade_tick: bool = False
    has_order_book_deltas: bool = False
    has_order_book_depths: bool = False
    is_backtest_ready: bool = False
    max_gap_seconds: float | None = None
    fenced_range_count: int = 0
    resync_count: int = 0
    desync_count: int = 0
    limitations: list[str] = Field(default_factory=list)


class QualityOffenderItem(StrictModel):
    """Legacy L2 quality offender (demoted, kept for optional depth10)."""
    instrument_id: str
    instrument_type: str
    quality_score: float = 100.0
    max_gap_seconds: float | None = None
    crossed_rate: float = 0.0
    empty_rate: float = 0.0
    bad_rate: float = 0.0
    snapshot_count: int = 0


class AuditSummary(StrictModel):
    instrument_count: int
    data_type_coverage: dict[str, int] = Field(default_factory=dict)
    total_row_counts: dict[str, int] = Field(default_factory=dict)
    # Deterministic-first readiness summary (primary)
    backtest_ready_count: int = 0
    consumable_count: int = 0
    avg_readiness_score: float = 0.0
    total_fenced_range_count: int = 0
    total_desync_count: int = 0
    total_resync_count: int = 0
    # Top offenders (readiness-first)
    top_readiness_offenders: list[ReadinessOffenderItem] = Field(default_factory=list)
    top_gap_offenders: list[ReadinessOffenderItem] = Field(default_factory=list)
    top_fenced_offenders: list[ReadinessOffenderItem] = Field(default_factory=list)
    # Legacy L2 summary fields (demoted)
    l2_sampled_snapshot_count: int = 0
    l2_bad_count: int = 0
    l2_bad_instrument_count: int = 0
    l2_bad_rate: float = 0.0
    overall_crossed_rate: float = 0.0
    overall_empty_rate: float = 0.0
    overall_monotonic_rate: float = 0.0
    overall_quality_score: float = 100.0
    session_break_count: int = 0
    chart_points: list[ChartPoint] = Field(default_factory=list)
    top_crossed_offenders: list[QualityOffenderItem] = Field(default_factory=list)
    top_empty_offenders: list[QualityOffenderItem] = Field(default_factory=list)


class AuditResponse(StrictModel):
    catalog_root: str
    generated_at: str
    cache_path: str | None = None
    inventory: InventoryResponse
    instruments: list[AuditInstrumentResult] = Field(default_factory=list)
    summary: AuditSummary
    warnings: list[CatalogWarning] = Field(default_factory=list)


class ProgressResponse(StrictModel):
    status: Literal["idle", "running", "completed", "failed"] = "idle"
    phase: str = "idle"
    message: str = "Audit has not been started yet."
    completed_steps: int = 0
    total_steps: int = 0
    percent: float = 0.0
    started_at: str | None = None
    finished_at: str | None = None
    cache_path: str | None = None
    error: str | None = None


# ── Search models ──────────────────────────────────────────────────────────

class InstrumentSearchItem(StrictModel):
    instrument_id: str
    instrument_type: str
    has_trade_tick: bool = False
    has_order_book_deltas: bool = False
    has_order_book_depths: bool = False


class InstrumentSearchResponse(StrictModel):
    catalog_root: str
    total: int = 0
    items: list[InstrumentSearchItem] = Field(default_factory=list)


# ── Coverage models ─────────────────────────────────────────────────────────

class CoverageSummary(StrictModel):
    data_type: str
    present: bool = False
    file_count: int = 0
    row_count: int = 0
    ts_event_min_ns: int | None = None
    ts_event_min_iso: str | None = None
    ts_event_max_ns: int | None = None
    ts_event_max_iso: str | None = None
    duration_seconds: float | None = None
    max_gap_ns: int | None = None
    max_gap_seconds: float | None = None
    missing_ratio_estimate: float = 0.0
    session_break_count: int = 0
    top_gaps: list[GapEntry] = Field(default_factory=list)
    error: str | None = None


class CoverageResponse(StrictModel):
    instrument_id: str
    instrument_type: str
    from_ns: int | None = None
    from_iso: str | None = None
    to_ns: int | None = None
    to_iso: str | None = None
    generated_at: str
    coverage: dict[str, CoverageSummary] = Field(default_factory=dict)
    warnings: list[CatalogWarning] = Field(default_factory=list)


# ── Trade models ────────────────────────────────────────────────────────────

class TradeSeriesPoint(StrictModel):
    ts_event_ns: int
    ts_event_iso: str
    price: float | None = None
    size: float | None = None
    aggressor_side: str | None = None
    trade_id: str | None = None
    trade_count: int | None = None
    volume: float | None = None
    avg_price: float | None = None
    last_price: float | None = None
    min_price: float | None = None
    max_price: float | None = None


class TradesResponse(StrictModel):
    instrument_id: str
    instrument_type: str
    mode: Literal["raw", "agg"] = "raw"
    bucket_s: int | None = None
    max_points: int = 10_000
    total_rows: int = 0
    returned_points: int = 0
    from_ns: int | None = None
    from_iso: str | None = None
    to_ns: int | None = None
    to_iso: str | None = None
    generated_at: str
    points: list[TradeSeriesPoint] = Field(default_factory=list)
    error: str | None = None


# ── Delta models (NEW — primary order book data) ──────────────────────────

class DeltaSeriesPoint(StrictModel):
    ts_event_ns: int
    ts_event_iso: str
    action: str | None = None  # ADD, UPDATE, DELETE, CLEAR
    side: str | None = None  # BID, ASK
    price: float | None = None
    size: float | None = None
    flags: int | None = None
    sequence: int | None = None
    # For aggregated mode
    delta_count: int | None = None
    add_count: int | None = None
    update_count: int | None = None
    delete_count: int | None = None
    clear_count: int | None = None


class DeltasSummaryResponse(StrictModel):
    instrument_id: str
    instrument_type: str
    generated_at: str
    present: bool = False
    file_count: int = 0
    total_rows: int = 0
    ts_event_min_ns: int | None = None
    ts_event_min_iso: str | None = None
    ts_event_max_ns: int | None = None
    ts_event_max_iso: str | None = None
    duration_seconds: float | None = None
    max_gap_seconds: float | None = None
    session_break_count: int = 0
    top_gaps: list[GapEntry] = Field(default_factory=list)
    error: str | None = None


class DeltasResponse(StrictModel):
    instrument_id: str
    instrument_type: str
    mode: Literal["raw", "agg"] = "raw"
    bucket_s: int | None = None
    max_points: int = 10_000
    total_rows: int = 0
    returned_points: int = 0
    from_ns: int | None = None
    from_iso: str | None = None
    to_ns: int | None = None
    to_iso: str | None = None
    generated_at: str
    points: list[DeltaSeriesPoint] = Field(default_factory=list)
    error: str | None = None


# ── L2 models (secondary — depth10 optional) ──────────────────────────────

class L2TimeseriesPoint(StrictModel):
    ts_event_ns: int
    ts_event_iso: str
    best_bid: float | None = None
    best_ask: float | None = None
    spread: float | None = None
    mid: float | None = None
    update_count: int | None = None
    update_rate_per_minute: float | None = None
    crossed_count: int | None = None
    empty_count: int | None = None
    bad_count: int | None = None
    is_crossed: bool | None = None
    is_sorted_ok: bool | None = None
    has_negative_qty: bool | None = None
    has_zero_qty: bool | None = None
    has_empty_side: bool | None = None


class L2TimeseriesResponse(StrictModel):
    instrument_id: str
    instrument_type: str
    mode: Literal["raw", "agg"] = "raw"
    bucket_s: int | None = None
    max_points: int = 10_000
    total_rows: int = 0
    returned_points: int = 0
    from_ns: int | None = None
    from_iso: str | None = None
    to_ns: int | None = None
    to_iso: str | None = None
    generated_at: str
    points: list[L2TimeseriesPoint] = Field(default_factory=list)
    error: str | None = None


class OrderBookLevel(StrictModel):
    level: int
    price: float
    size: float


class L2SnapshotSummary(StrictModel):
    index: int
    ts_event_ns: int
    ts_event_iso: str
    best_bid: float | None = None
    best_ask: float | None = None
    spread: float | None = None
    mid: float | None = None
    is_crossed: bool = False
    is_sorted_ok: bool = True
    has_negative_qty: bool = False
    has_zero_qty: bool = False
    has_empty_side: bool = False
    issues: list[str] = Field(default_factory=list)


class L2Snapshot(L2SnapshotSummary):
    bids: list[OrderBookLevel] = Field(default_factory=list)
    asks: list[OrderBookLevel] = Field(default_factory=list)
    bid_sizes: list[float] = Field(default_factory=list)
    ask_sizes: list[float] = Field(default_factory=list)
    flags: int | None = None
    sequence: int | None = None


class L2SnapshotResponse(StrictModel):
    instrument_id: str
    instrument_type: str
    total_snapshots: int = 0
    resolved_index: int | None = None
    requested_index: int | None = None
    requested_ts_ns: int | None = None
    requested_ts_iso: str | None = None
    generated_at: str
    snapshot: L2Snapshot | None = None
    context: list[L2SnapshotSummary] = Field(default_factory=list)
    error: str | None = None


class QualitySnapshotIssue(StrictModel):
    index: int
    ts_event_ns: int
    ts_event_iso: str
    issues: list[str] = Field(default_factory=list)
    best_bid: float | None = None
    best_ask: float | None = None
    spread: float | None = None
    mid: float | None = None


class L2QualityResponse(StrictModel):
    """Optional L2 depth10 quality response (secondary)."""
    instrument_id: str
    instrument_type: str
    from_ns: int | None = None
    from_iso: str | None = None
    to_ns: int | None = None
    to_iso: str | None = None
    generated_at: str
    snapshot_count: int = 0
    crossed_count: int = 0
    crossed_rate: float = 0.0
    monotonic_violation_count: int = 0
    monotonic_violation_rate: float = 0.0
    negative_qty_count: int = 0
    zero_qty_count: int = 0
    empty_side_count: int = 0
    empty_side_rate: float = 0.0
    bad_snapshot_count: int = 0
    bad_snapshot_rate: float = 0.0
    quality_score: float = 100.0
    max_gap_seconds: float | None = None
    session_break_count: int = 0
    session_break_threshold_s: int = 300
    top_gaps: list[GapEntry] = Field(default_factory=list)
    bad_snapshots: list[QualitySnapshotIssue] = Field(default_factory=list)
    error: str | None = None
