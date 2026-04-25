"""Tests for the deterministic-first Nautilus Catalog Viewer.

Covers:
- Inventory with deltas but no depth10
- Inventory with trades + deltas + optional depth10
- Deterministic report ingestion / fenced range parsing
- Delta-first readiness scoring
- Instrument page behaviour when depth10 is absent
- Export bundle contents for delta-first datasets
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from app.catalog_scan import (
    CatalogScanner,
    EVENT_DATA_TYPES,
    _load_report_context,
    compute_readiness_score,
)
from app.models import (
    AuditInstrumentResult,
    DataTypeAuditStats,
    DeltasResponse,
    DeltasSummaryResponse,
    FencedRange,
    ReadinessOffenderItem,
    ReadinessResponse,
    ReadinessResult,
    ReportContext,
    ResyncEvent,
    SessionBoundary,
)


# ── Fixtures ────────────────────────────────────────────────────────────────

def _write_parquet(path: Path, ts_values: list[int], extra_columns: dict[str, list[Any]] | None = None) -> Path:
    """Create a minimal parquet file with ts_event column."""
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = {"ts_event": pa.array(ts_values, type=pa.int64())}
    if extra_columns:
        for name, values in extra_columns.items():
            if isinstance(values[0], (int, np.integer)):
                columns[name] = pa.array(values, type=pa.int64())
            elif isinstance(values[0], float):
                columns[name] = pa.array(values, type=pa.float64())
            elif isinstance(values[0], bytes):
                columns[name] = pa.array(values, type=pa.binary())
            else:
                columns[name] = pa.array(values, type=pa.string())
    table = pa.table(columns)
    pq.write_table(table, str(path))
    return path


@pytest.fixture()
def catalog_with_deltas_only(tmp_path: Path) -> Path:
    """Catalog with trade_tick + order_book_deltas but NO depth10."""
    root = tmp_path / "catalog"
    instrument_id = "BTCUSDT-PERP.BINANCE"

    # trade_tick data
    _write_parquet(
        root / "data" / "trade_tick" / instrument_id / "part-0.parquet",
        ts_values=[1_000_000_000, 2_000_000_000, 3_000_000_000, 4_000_000_000],
        extra_columns={
            "price": [b"\x00" * 16] * 4,
            "size": [b"\x00" * 16] * 4,
            "aggressor_side": [1, 2, 1, 2],
        },
    )

    # order_book_deltas data
    _write_parquet(
        root / "data" / "order_book_deltas" / instrument_id / "part-0.parquet",
        ts_values=[1_000_000_000, 1_500_000_000, 2_000_000_000, 2_500_000_000, 3_000_000_000],
        extra_columns={
            "action": [1, 2, 1, 3, 1],
            "side": [1, 2, 1, 2, 1],
            "price": [b"\x00" * 16] * 5,
            "size": [b"\x00" * 16] * 5,
            "flags": [0, 0, 0, 0, 0],
            "sequence": [1, 2, 3, 4, 5],
        },
    )

    # instrument type directory
    (root / "data" / "crypto_perpetual" / instrument_id).mkdir(parents=True, exist_ok=True)

    return root


@pytest.fixture()
def catalog_with_all_types(tmp_path: Path) -> Path:
    """Catalog with trade_tick + order_book_deltas + order_book_depths (all three)."""
    root = tmp_path / "catalog"
    instrument_id = "ETHUSDT.BINANCE"

    # trade_tick data
    _write_parquet(
        root / "data" / "trade_tick" / instrument_id / "part-0.parquet",
        ts_values=[1_000_000_000, 2_000_000_000, 3_000_000_000],
        extra_columns={
            "price": [b"\x00" * 16] * 3,
            "size": [b"\x00" * 16] * 3,
            "aggressor_side": [1, 2, 1],
        },
    )

    # order_book_deltas
    _write_parquet(
        root / "data" / "order_book_deltas" / instrument_id / "part-0.parquet",
        ts_values=[1_000_000_000, 2_000_000_000, 3_000_000_000],
        extra_columns={
            "action": [1, 2, 1],
            "side": [1, 2, 1],
            "price": [b"\x00" * 16] * 3,
            "size": [b"\x00" * 16] * 3,
            "flags": [0, 0, 0],
            "sequence": [1, 2, 3],
        },
    )

    # order_book_depths (depth10 -- optional)
    depth_cols: dict[str, list[Any]] = {}
    for prefix in ("bid_price_", "ask_price_", "bid_size_", "ask_size_"):
        for level in range(10):
            depth_cols[f"{prefix}{level}"] = [b"\x00" * 16] * 3
    depth_cols["flags"] = [0, 0, 0]
    depth_cols["sequence"] = [1, 2, 3]
    _write_parquet(
        root / "data" / "order_book_depths" / instrument_id / "part-0.parquet",
        ts_values=[1_000_000_000, 2_000_000_000, 3_000_000_000],
        extra_columns=depth_cols,
    )

    # instrument type directory
    (root / "data" / "currency_pair" / instrument_id).mkdir(parents=True, exist_ok=True)

    return root


@pytest.fixture()
def catalog_with_report(catalog_with_deltas_only: Path) -> Path:
    """Catalog with a deterministic report for the instrument."""
    root = catalog_with_deltas_only
    instrument_id = "BTCUSDT-PERP.BINANCE"
    report_dir = root / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)

    report = {
        "snapshot_seed_count": 3,
        "resync_count": 2,
        "desync_count": 1,
        "fenced_ranges": [
            {
                "start_ns": 1_500_000_000,
                "start_iso": "1970-01-01T00:00:01.500000000Z",
                "end_ns": 2_000_000_000,
                "end_iso": "1970-01-01T00:00:02.000000000Z",
                "reason": "Exchange maintenance window",
            },
        ],
        "session_boundaries": [
            {"ts_ns": 1_000_000_000, "ts_iso": "1970-01-01T00:00:01.000000000Z", "kind": "start", "label": "Session 1"},
            {"ts_ns": 4_000_000_000, "ts_iso": "1970-01-01T00:00:04.000000000Z", "kind": "end", "label": "Session 1"},
        ],
        "resync_events": [
            {"ts_ns": 1_200_000_000, "ts_iso": "1970-01-01T00:00:01.200000000Z", "kind": "snapshot_seed", "detail": "Initial seed"},
            {"ts_ns": 1_800_000_000, "ts_iso": "1970-01-01T00:00:01.800000000Z", "kind": "resync", "detail": "Gap detected"},
            {"ts_ns": 2_200_000_000, "ts_iso": "1970-01-01T00:00:02.200000000Z", "kind": "desync", "detail": "Sequence break"},
        ],
        "last_committed_update_id": "update-12345",
        "trade_id_diagnostics": ["Duplicate trade_id at ts=1500000000"],
        "converter_warnings": ["Slow converter throughput detected"],
    }
    (report_dir / f"{instrument_id}.json").write_text(json.dumps(report), encoding="utf-8")

    return root


# ── Tests: EVENT_DATA_TYPES ────────────────────────────────────────────────

def test_event_data_types_include_deltas():
    """order_book_deltas must be in the EVENT_DATA_TYPES tuple."""
    assert "trade_tick" in EVENT_DATA_TYPES
    assert "order_book_deltas" in EVENT_DATA_TYPES
    assert "order_book_depths" in EVENT_DATA_TYPES


# ── Tests: Inventory with deltas but no depth10 ───────────────────────────

def test_inventory_deltas_only(catalog_with_deltas_only: Path):
    """Inventory should correctly detect deltas when depth10 is absent."""
    scanner = CatalogScanner(catalog_with_deltas_only)
    inventory = scanner.scan_inventory()

    assert len(inventory.instruments) >= 1
    instrument = next(i for i in inventory.instruments if "BTCUSDT-PERP" in i.instrument_id)

    assert instrument.coverage["trade_tick"].present is True
    assert instrument.coverage["order_book_deltas"].present is True
    assert instrument.coverage["order_book_depths"].present is False
    assert instrument.has_any_data is True


def test_inventory_all_types(catalog_with_all_types: Path):
    """Inventory should detect all three data types when all present."""
    scanner = CatalogScanner(catalog_with_all_types)
    inventory = scanner.scan_inventory()

    instrument = next(i for i in inventory.instruments if "ETHUSDT" in i.instrument_id)

    assert instrument.coverage["trade_tick"].present is True
    assert instrument.coverage["order_book_deltas"].present is True
    assert instrument.coverage["order_book_depths"].present is True


# ── Tests: Deterministic report ingestion ──────────────────────────────────

def test_report_loading(catalog_with_report: Path):
    """Report context should load fenced ranges, session boundaries, resync events."""
    report = _load_report_context(catalog_with_report / "reports", "BTCUSDT-PERP.BINANCE")

    assert report.report_found is True
    assert report.snapshot_seed_count == 3
    assert report.resync_count == 2
    assert report.desync_count == 1
    assert len(report.fenced_ranges) == 1
    assert report.fenced_ranges[0].reason == "Exchange maintenance window"
    assert len(report.session_boundaries) == 2
    assert len(report.resync_events) == 3
    assert report.last_committed_update_id == "update-12345"
    assert len(report.trade_id_diagnostics) == 1
    assert len(report.converter_warnings) == 1


def test_report_missing(tmp_path: Path):
    """Missing report should return report_found=False."""
    report = _load_report_context(tmp_path / "reports", "NONEXIST.BINANCE")
    assert report.report_found is False
    assert report.snapshot_seed_count == 0


def test_report_invalid_json(tmp_path: Path):
    """Invalid JSON should return report_found=False gracefully."""
    report_dir = tmp_path / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "BAD.json").write_text("not json", encoding="utf-8")
    report = _load_report_context(report_dir, "BAD")
    assert report.report_found is False


# ── Tests: Fenced range parsing ────────────────────────────────────────────

def test_fenced_range_model():
    """FencedRange model should parse correctly."""
    fr = FencedRange(
        start_ns=1_000_000_000,
        start_iso="1970-01-01T00:00:01.000000000Z",
        end_ns=2_000_000_000,
        end_iso="1970-01-01T00:00:02.000000000Z",
        reason="test reason",
    )
    assert fr.start_ns == 1_000_000_000
    assert fr.reason == "test reason"


# ── Tests: Readiness scoring ──────────────────────────────────────────────

def test_readiness_score_full():
    """Full data availability should yield a high readiness score."""
    score = compute_readiness_score(
        has_trade_tick=True,
        has_order_book_deltas=True,
        has_order_book_depths=True,
        trade_row_count=10_000,
        delta_row_count=50_000,
        trade_max_gap_seconds=None,
        delta_max_gap_seconds=None,
        fenced_range_count=0,
        desync_count=0,
        resync_count=0,
        session_break_count=0,
    )
    assert score >= 70.0  # 20 + 25 + 5 + 10 + 10 = 70 base


def test_readiness_score_no_deltas():
    """Missing deltas should significantly lower the score."""
    score = compute_readiness_score(
        has_trade_tick=True,
        has_order_book_deltas=False,
        has_order_book_depths=False,
        trade_row_count=10_000,
        delta_row_count=0,
        trade_max_gap_seconds=None,
        delta_max_gap_seconds=None,
        fenced_range_count=0,
        desync_count=0,
        resync_count=0,
        session_break_count=0,
    )
    # Only trades: 20 + 10 = 30
    assert score <= 35.0


def test_readiness_score_with_penalties():
    """Fenced ranges and desyncs should apply penalties."""
    score = compute_readiness_score(
        has_trade_tick=True,
        has_order_book_deltas=True,
        has_order_book_depths=False,
        trade_row_count=10_000,
        delta_row_count=50_000,
        trade_max_gap_seconds=100.0,
        delta_max_gap_seconds=50.0,
        fenced_range_count=3,
        desync_count=2,
        resync_count=10,
        session_break_count=5,
    )
    # Base 65, minus gap/fenced/desync/resync penalties
    assert score < 60.0
    assert score >= 0.0


def test_readiness_score_empty():
    """No data at all should give 0."""
    score = compute_readiness_score(
        has_trade_tick=False,
        has_order_book_deltas=False,
        has_order_book_depths=False,
        trade_row_count=0,
        delta_row_count=0,
        trade_max_gap_seconds=None,
        delta_max_gap_seconds=None,
        fenced_range_count=0,
        desync_count=0,
        resync_count=0,
        session_break_count=0,
    )
    assert score == 0.0


# ── Tests: ReadinessResult model ──────────────────────────────────────────

def test_readiness_result_backtest_ready():
    """is_backtest_ready should be True when trades + deltas + no desyncs."""
    r = ReadinessResult(
        instrument_id="TEST",
        instrument_type="currency_pair",
        has_trade_tick=True,
        has_order_book_deltas=True,
        has_order_book_depths=False,
        is_consumable=True,
        is_backtest_ready=True,
        readiness_score=65.0,
    )
    assert r.is_backtest_ready is True
    assert r.delta_first_only is False  # needs has_deltas=True and not has_depths for True


def test_readiness_result_delta_first_only():
    """delta_first_only should be True when deltas present but no depth10."""
    r = ReadinessResult(
        instrument_id="TEST",
        instrument_type="currency_pair",
        has_trade_tick=True,
        has_order_book_deltas=True,
        has_order_book_depths=False,
        delta_first_only=True,
    )
    assert r.delta_first_only is True


# ── Tests: Instrument page when depth10 is absent ─────────────────────────

def test_instrument_coverage_no_depth10(catalog_with_deltas_only: Path):
    """Coverage query should work and show depth as absent gracefully."""
    scanner = CatalogScanner(catalog_with_deltas_only)
    qs = scanner.query_service

    coverage = qs.get_coverage("BTCUSDT-PERP.BINANCE")
    assert "trade_tick" in coverage.coverage
    assert "order_book_deltas" in coverage.coverage
    assert "order_book_depths" in coverage.coverage
    assert coverage.coverage["trade_tick"].present is True
    assert coverage.coverage["order_book_deltas"].present is True
    assert coverage.coverage["order_book_depths"].present is False


# ── Tests: Delta query support ────────────────────────────────────────────

def test_deltas_summary(catalog_with_deltas_only: Path):
    """Deltas summary should return row counts and time range."""
    scanner = CatalogScanner(catalog_with_deltas_only)
    qs = scanner.query_service

    summary = qs.get_deltas_summary("BTCUSDT-PERP.BINANCE")
    assert summary.present is True
    assert summary.total_rows == 5
    assert summary.file_count == 1
    assert summary.ts_event_min_ns is not None
    assert summary.ts_event_max_ns is not None


def test_deltas_query_raw(catalog_with_deltas_only: Path):
    """Raw delta query should return individual events."""
    scanner = CatalogScanner(catalog_with_deltas_only)
    qs = scanner.query_service

    response = qs.get_deltas("BTCUSDT-PERP.BINANCE", mode="raw")
    assert response.total_rows == 5
    assert response.returned_points == 5
    assert len(response.points) == 5


def test_deltas_query_agg(catalog_with_deltas_only: Path):
    """Aggregated delta query should bucket events."""
    scanner = CatalogScanner(catalog_with_deltas_only)
    qs = scanner.query_service

    response = qs.get_deltas("BTCUSDT-PERP.BINANCE", mode="agg", bucket_s=1)
    assert response.mode == "agg"
    assert response.returned_points >= 1


def test_deltas_query_nonexistent(catalog_with_deltas_only: Path):
    """Delta query for nonexistent instrument should return empty."""
    scanner = CatalogScanner(catalog_with_deltas_only)
    qs = scanner.query_service

    response = qs.get_deltas("NONEXIST.BINANCE", mode="raw")
    assert response.total_rows == 0


# ── Tests: Audit with readiness ───────────────────────────────────────────

def test_audit_includes_readiness(catalog_with_deltas_only: Path):
    """Audit results should include readiness for each instrument."""
    scanner = CatalogScanner(catalog_with_deltas_only)
    audit = scanner.run_audit()

    assert audit.summary.instrument_count >= 1
    assert "order_book_deltas" in audit.summary.data_type_coverage

    instrument = next(i for i in audit.instruments if "BTCUSDT-PERP" in i.instrument_id)
    r = instrument.readiness
    assert r.has_trade_tick is True
    assert r.has_order_book_deltas is True
    assert r.has_order_book_depths is False
    assert r.is_consumable is True
    assert r.readiness_score > 0


def test_audit_with_report(catalog_with_report: Path):
    """Audit should incorporate report context into readiness."""
    scanner = CatalogScanner(catalog_with_report)
    audit = scanner.run_audit()

    instrument = next(i for i in audit.instruments if "BTCUSDT-PERP" in i.instrument_id)
    r = instrument.readiness
    assert r.report.report_found is True
    assert r.fenced_range_count == 1
    assert r.desync_count == 1
    assert r.snapshot_seed_count == 3
    assert r.is_backtest_ready is False  # desync > 0


def test_audit_summary_readiness_fields(catalog_with_deltas_only: Path):
    """Audit summary should have deterministic-first readiness fields."""
    scanner = CatalogScanner(catalog_with_deltas_only)
    audit = scanner.run_audit()

    summary = audit.summary
    assert hasattr(summary, "backtest_ready_count")
    assert hasattr(summary, "consumable_count")
    assert hasattr(summary, "avg_readiness_score")
    assert hasattr(summary, "total_fenced_range_count")
    assert hasattr(summary, "total_desync_count")
    assert hasattr(summary, "top_readiness_offenders")
    assert isinstance(summary.top_readiness_offenders, list)


# ── Tests: Export bundle for delta-first datasets ─────────────────────────

def test_export_bundle_delta_first(catalog_with_deltas_only: Path):
    """Debug bundle should include deltas summary and readiness."""
    scanner = CatalogScanner(catalog_with_deltas_only)
    qs = scanner.query_service

    raw_bytes = qs.export_bundle_json("BTCUSDT-PERP.BINANCE")
    bundle = json.loads(raw_bytes)

    assert "instrument_id" in bundle
    assert "coverage" in bundle
    assert "trades" in bundle
    assert "deltas_summary" in bundle
    assert "readiness" in bundle
    assert bundle["readiness"]["readiness"]["has_order_book_deltas"] is True
    assert bundle["readiness"]["readiness"]["has_order_book_depths"] is False


# ── Tests: Model backward compat ──────────────────────────────────────────

def test_audit_instrument_result_has_both_readiness_and_legacy():
    """AuditInstrumentResult should have both readiness (primary) and legacy quality fields."""
    result = AuditInstrumentResult(
        instrument_id="TEST",
        instrument_type="currency_pair",
    )
    assert hasattr(result, "readiness")
    assert hasattr(result, "quality_score")
    assert hasattr(result, "l2_check")
    assert result.readiness.readiness_score == 0.0
    assert result.quality_score == 100.0
