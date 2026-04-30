"""Tests for TradeTick audit / catalog-reading logic.

Covers:
1. TradeTick present for a spot instrument
2. TradeTick present for a futures/perpetual instrument
3. TradeTick absent genuinely (no files)
4. Parquet exists but bad/missing schema -> corrupt=True, present=True, not absent
5. Audit does not confuse order_book_deltas with trade_tick
6. Audit correctly reports row_count > 0 and non-null ts_event range
7. Converter report cross-check: instruments_with_trades match -> no mismatch
8. Converter report cross-check: count mismatch -> trade_tick_detected_mismatch=True + warning
9. debug_trade_tick() returns correct fields
10. Schema tolerance: ts field named differently (event_ts / ts_init fallback)
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

from app.catalog_scan import CatalogScanner, EVENT_DATA_TYPES


# ── Helpers ─────────────────────────────────────────────────────────────────

def _ts(offset_s: int) -> int:
    """Nanosecond timestamp starting at a recognizable base epoch."""
    return 1_777_000_000_000_000_000 + offset_s * 1_000_000_000


def _write_parquet(path: Path, ts_values: list[int], extra_columns: dict[str, list[Any]] | None = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns: dict[str, pa.Array] = {"ts_event": pa.array(ts_values, type=pa.int64())}
    if extra_columns:
        for name, values in extra_columns.items():
            if not values:
                continue
            first = values[0]
            if isinstance(first, (int, np.integer)):
                columns[name] = pa.array(values, type=pa.int64())
            elif isinstance(first, float):
                columns[name] = pa.array(values, type=pa.float64())
            elif isinstance(first, bytes):
                columns[name] = pa.array(values, type=pa.binary())
            else:
                columns[name] = pa.array(values, type=pa.string())
    pq.write_table(pa.table(columns), str(path))
    return path


def _write_trade_parquet(path: Path, ts_values: list[int]) -> Path:
    return _write_parquet(
        path,
        ts_values=ts_values,
        extra_columns={
            "price": [b"\x00" * 16] * len(ts_values),
            "size": [b"\x00" * 16] * len(ts_values),
            "aggressor_side": [1 if i % 2 == 0 else 2 for i in range(len(ts_values))],
            "trade_id": [str(i) for i in range(len(ts_values))],
        },
    )


def _write_delta_parquet(path: Path, ts_values: list[int]) -> Path:
    return _write_parquet(
        path,
        ts_values=ts_values,
        extra_columns={
            "action": [1] * len(ts_values),
            "side": [1] * len(ts_values),
            "price": [b"\x00" * 16] * len(ts_values),
            "size": [b"\x00" * 16] * len(ts_values),
            "flags": [0] * len(ts_values),
            "sequence": list(range(1, len(ts_values) + 1)),
        },
    )


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture()
def catalog_spot_with_trades(tmp_path: Path) -> Path:
    root = tmp_path / "catalog"
    iid = "BTCUSDT.BINANCE"
    _write_trade_parquet(root / "data" / "trade_tick" / iid / "part-0.parquet", [_ts(0), _ts(1), _ts(2), _ts(3), _ts(4)])
    _write_delta_parquet(root / "data" / "order_book_deltas" / iid / "part-0.parquet", [_ts(0), _ts(1), _ts(2)])
    (root / "data" / "currency_pair" / iid).mkdir(parents=True, exist_ok=True)
    return root


@pytest.fixture()
def catalog_perp_with_trades(tmp_path: Path) -> Path:
    """A perpetual/futures instrument that genuinely DOES have trade_tick data."""
    root = tmp_path / "catalog"
    iid = "BTCUSDT-PERP.BINANCE"
    _write_trade_parquet(root / "data" / "trade_tick" / iid / "part-0.parquet", [_ts(0), _ts(2), _ts(4)])
    _write_delta_parquet(root / "data" / "order_book_deltas" / iid / "part-0.parquet", [_ts(0), _ts(1), _ts(2)])
    (root / "data" / "crypto_perpetual" / iid).mkdir(parents=True, exist_ok=True)
    return root


@pytest.fixture()
def catalog_perp_no_trades(tmp_path: Path) -> Path:
    """A perpetual instrument with order_book_deltas but NO trade_tick — genuine absence."""
    root = tmp_path / "catalog"
    iid = "ETHUSDT-PERP.BINANCE"
    _write_delta_parquet(root / "data" / "order_book_deltas" / iid / "part-0.parquet", [_ts(0), _ts(1), _ts(2)])
    (root / "data" / "crypto_perpetual" / iid).mkdir(parents=True, exist_ok=True)
    return root


@pytest.fixture()
def catalog_with_corrupt_trade_parquet(tmp_path: Path) -> Path:
    """An instrument where trade_tick directory exists but the parquet has no ts_event column."""
    root = tmp_path / "catalog"
    iid = "BADSPOT.BINANCE"
    # Write a parquet with wrong schema — no ts_event column at all
    parquet_path = root / "data" / "trade_tick" / iid / "part-0.parquet"
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.table({"garbage_col": pa.array([1, 2, 3], type=pa.int32())})
    pq.write_table(table, str(parquet_path))
    (root / "data" / "currency_pair" / iid).mkdir(parents=True, exist_ok=True)
    return root


@pytest.fixture()
def catalog_mixed(tmp_path: Path) -> Path:
    """Catalog with one spot instrument (has trades) and one perpetual (no trades)."""
    root = tmp_path / "catalog"

    spot = "SOLUSDT.BINANCE"
    _write_trade_parquet(root / "data" / "trade_tick" / spot / "part-0.parquet", [_ts(0), _ts(1), _ts(2)])
    _write_delta_parquet(root / "data" / "order_book_deltas" / spot / "part-0.parquet", [_ts(0), _ts(1)])
    (root / "data" / "currency_pair" / spot).mkdir(parents=True, exist_ok=True)

    perp = "SOLUSDT-PERP.BINANCE"
    _write_delta_parquet(root / "data" / "order_book_deltas" / perp / "part-0.parquet", [_ts(0), _ts(1)])
    (root / "data" / "crypto_perpetual" / perp).mkdir(parents=True, exist_ok=True)

    return root


def _write_converter_report(root: Path, instruments_with_trades: int, no_data_list: list[str] | None = None) -> Path:
    report_dir = root / "converter_reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "date": "2026-04-24",
        "status": "ok",
        "total_trades_written": 1000,
        "total_order_book_deltas_written": 5000,
        "total_depth10_written": 500,
        "bad_lines": 0,
        "data_presence": {
            "instruments_defined": instruments_with_trades + len(no_data_list or []),
            "instruments_with_trades": instruments_with_trades,
            "instruments_with_depth": instruments_with_trades + len(no_data_list or []),
            "instruments_with_both": instruments_with_trades,
            "instruments_with_no_data": len(no_data_list or []),
            "no_data_list": no_data_list or [],
        },
        "per_symbol_fenced_ranges": {},
    }
    report_path = report_dir / "2026-04-24.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    return report_dir


# ── Tests: 1. TradeTick present for spot instrument ──────────────────────────

def test_trade_tick_present_spot(catalog_spot_with_trades: Path) -> None:
    """Spot instrument with trade parquet files must be detected as present=True."""
    scanner = CatalogScanner(catalog_spot_with_trades)
    inventory = scanner.scan_inventory()
    inst = next(i for i in inventory.instruments if "BTCUSDT.BINANCE" == i.instrument_id)

    assert inst.coverage["trade_tick"].present is True
    assert inst.coverage["trade_tick"].file_count == 1


def test_trade_tick_stats_spot(catalog_spot_with_trades: Path) -> None:
    """Spot instrument audit must have row_count > 0 and non-null ts_event_min/max."""
    scanner = CatalogScanner(catalog_spot_with_trades)
    audit = scanner.run_audit()
    inst = next(i for i in audit.instruments if "BTCUSDT.BINANCE" == i.instrument_id)
    tt = inst.data_types["trade_tick"]

    assert tt.present is True
    assert tt.row_count_estimate == 5
    assert tt.ts_event_min_ns is not None
    assert tt.ts_event_max_ns is not None
    assert tt.ts_event_min_ns < tt.ts_event_max_ns
    assert tt.ts_event_min_ns == _ts(0)
    assert tt.ts_event_max_ns == _ts(4)


# ── Tests: 2. TradeTick present for futures/perpetual instrument ──────────────

def test_trade_tick_present_perp(catalog_perp_with_trades: Path) -> None:
    """Perpetual instrument WITH trade data must be detected present=True (not silently absent)."""
    scanner = CatalogScanner(catalog_perp_with_trades)
    inventory = scanner.scan_inventory()
    inst = next(i for i in inventory.instruments if "BTCUSDT-PERP.BINANCE" == i.instrument_id)

    assert inst.coverage["trade_tick"].present is True


def test_trade_tick_stats_perp(catalog_perp_with_trades: Path) -> None:
    """Perpetual instrument audit must have correct stats when trade data present."""
    scanner = CatalogScanner(catalog_perp_with_trades)
    audit = scanner.run_audit()
    inst = next(i for i in audit.instruments if "BTCUSDT-PERP.BINANCE" == i.instrument_id)
    tt = inst.data_types["trade_tick"]

    assert tt.present is True
    assert tt.row_count_estimate == 3
    assert tt.ts_event_min_ns == _ts(0)
    assert tt.ts_event_max_ns == _ts(4)


# ── Tests: 3. TradeTick genuinely absent ──────────────────────────────────────

def test_trade_tick_absent_perp_no_trades(catalog_perp_no_trades: Path) -> None:
    """Perpetual instrument with NO trade files must show present=False (not corrupt or error)."""
    scanner = CatalogScanner(catalog_perp_no_trades)
    inventory = scanner.scan_inventory()
    inst = next(i for i in inventory.instruments if "ETHUSDT-PERP.BINANCE" == i.instrument_id)

    assert inst.coverage["trade_tick"].present is False


def test_trade_tick_absent_no_corrupt_flag(catalog_perp_no_trades: Path) -> None:
    """Genuinely absent trade_tick must NOT be marked corrupt."""
    scanner = CatalogScanner(catalog_perp_no_trades)
    audit = scanner.run_audit()
    inst = next(i for i in audit.instruments if "ETHUSDT-PERP.BINANCE" == i.instrument_id)
    tt = inst.data_types["trade_tick"]

    assert tt.present is False
    assert tt.corrupt is False
    assert tt.error is None


# ── Tests: 4. Parquet exists but schema error -> corrupt, not absent ──────────

def test_trade_tick_corrupt_schema_present_not_absent(catalog_with_corrupt_trade_parquet: Path) -> None:
    """Parquet files exist but have wrong schema -> present=True, corrupt=True (not absent)."""
    scanner = CatalogScanner(catalog_with_corrupt_trade_parquet)
    inventory = scanner.scan_inventory()
    inst = next(i for i in inventory.instruments if "BADSPOT.BINANCE" == i.instrument_id)

    # File exists, so inventory must say present
    assert inst.coverage["trade_tick"].present is True


def test_trade_tick_corrupt_schema_marks_error(catalog_with_corrupt_trade_parquet: Path) -> None:
    """Parquet with missing ts_event column -> audit marks corrupt=True and records error, not absent."""
    scanner = CatalogScanner(catalog_with_corrupt_trade_parquet)
    audit = scanner.run_audit()
    inst = next(i for i in audit.instruments if "BADSPOT.BINANCE" == i.instrument_id)
    tt = inst.data_types["trade_tick"]

    assert tt.present is True              # file exists -> present
    assert tt.corrupt is True              # but schema is unreadable
    assert tt.error is not None            # error message recorded
    # ts bounds and row_count may be 0 / None due to error — that is acceptable
    # The important thing is present=True, not absent


# ── Tests: 5. Audit does not confuse order_book_deltas with trade_tick ────────

def test_audit_no_confusion_order_book_vs_trade(catalog_perp_no_trades: Path) -> None:
    """An instrument with ONLY order_book_deltas must NOT have trade_tick marked present."""
    scanner = CatalogScanner(catalog_perp_no_trades)
    audit = scanner.run_audit()
    inst = next(i for i in audit.instruments if "ETHUSDT-PERP.BINANCE" == i.instrument_id)

    assert inst.data_types["order_book_deltas"].present is True     # has deltas
    assert inst.data_types["trade_tick"].present is False           # no trades
    assert inst.data_types["order_book_depths"].present is False    # no depths


def test_audit_trade_and_deltas_independent(catalog_mixed: Path) -> None:
    """Spot has both; perp has only deltas. They must not interfere."""
    scanner = CatalogScanner(catalog_mixed)
    audit = scanner.run_audit()

    spot = next(i for i in audit.instruments if i.instrument_id == "SOLUSDT.BINANCE")
    perp = next(i for i in audit.instruments if i.instrument_id == "SOLUSDT-PERP.BINANCE")

    assert spot.data_types["trade_tick"].present is True
    assert spot.data_types["order_book_deltas"].present is True

    assert perp.data_types["trade_tick"].present is False
    assert perp.data_types["order_book_deltas"].present is True


# ── Tests: 6. row_count > 0 and non-null ts_event range ──────────────────────

def test_audit_trade_tick_row_count_nonzero(catalog_spot_with_trades: Path) -> None:
    """row_count_estimate must be > 0 for instruments with trade data."""
    scanner = CatalogScanner(catalog_spot_with_trades)
    audit = scanner.run_audit()
    inst = next(i for i in audit.instruments if i.instrument_id == "BTCUSDT.BINANCE")

    assert inst.data_types["trade_tick"].row_count_estimate > 0


def test_audit_trade_tick_ts_range_non_null(catalog_spot_with_trades: Path) -> None:
    """ts_event_min_ns and ts_event_max_ns must both be non-null for instruments with data."""
    scanner = CatalogScanner(catalog_spot_with_trades)
    audit = scanner.run_audit()
    inst = next(i for i in audit.instruments if i.instrument_id == "BTCUSDT.BINANCE")
    tt = inst.data_types["trade_tick"]

    assert tt.ts_event_min_ns is not None
    assert tt.ts_event_max_ns is not None
    assert tt.ts_event_min_iso is not None
    assert tt.ts_event_max_iso is not None


# ── Tests: 7. Converter cross-check: no mismatch ─────────────────────────────

def test_converter_cross_check_no_mismatch(tmp_path: Path) -> None:
    """When converter instruments_with_trades matches viewer count, no mismatch warning."""
    root = tmp_path / "catalog"
    iid = "BTCUSDT.BINANCE"
    _write_trade_parquet(root / "data" / "trade_tick" / iid / "part-0.parquet", [_ts(0), _ts(1)])
    _write_delta_parquet(root / "data" / "order_book_deltas" / iid / "part-0.parquet", [_ts(0)])
    (root / "data" / "currency_pair" / iid).mkdir(parents=True, exist_ok=True)

    report_dir = _write_converter_report(root, instruments_with_trades=1, no_data_list=[])
    scanner = CatalogScanner(root, converter_reports_dir=report_dir)
    audit = scanner.run_audit()

    assert audit.summary.trade_tick_detected_mismatch is False
    assert audit.summary.converter_trade_tick_instrument_count == 1
    assert audit.summary.viewer_trade_tick_instrument_count == 1
    mismatch_warnings = [w for w in audit.warnings if w.code == "trade_tick_detected_mismatch"]
    assert len(mismatch_warnings) == 0


# ── Tests: 8. Converter cross-check: mismatch detected ───────────────────────

def test_converter_cross_check_mismatch_detected(tmp_path: Path) -> None:
    """When converter reports more trades than viewer finds, mismatch is flagged."""
    root = tmp_path / "catalog"
    iid = "BTCUSDT.BINANCE"
    _write_trade_parquet(root / "data" / "trade_tick" / iid / "part-0.parquet", [_ts(0), _ts(1)])
    _write_delta_parquet(root / "data" / "order_book_deltas" / iid / "part-0.parquet", [_ts(0)])
    (root / "data" / "currency_pair" / iid).mkdir(parents=True, exist_ok=True)

    # Converter says 5 instruments have trades, viewer only sees 1
    report_dir = _write_converter_report(root, instruments_with_trades=5, no_data_list=["MISSING.BINANCE"])
    scanner = CatalogScanner(root, converter_reports_dir=report_dir)
    audit = scanner.run_audit()

    assert audit.summary.trade_tick_detected_mismatch is True
    assert audit.summary.converter_trade_tick_instrument_count == 5
    assert audit.summary.viewer_trade_tick_instrument_count == 1
    mismatch_warnings = [w for w in audit.warnings if w.code == "trade_tick_detected_mismatch"]
    assert len(mismatch_warnings) == 1
    assert "5" in mismatch_warnings[0].message
    assert "1" in mismatch_warnings[0].message


def test_converter_no_report_no_mismatch_field(catalog_spot_with_trades: Path) -> None:
    """When no converter report is configured, cross-check fields are None/False, no error."""
    scanner = CatalogScanner(catalog_spot_with_trades)  # no converter_reports_dir
    audit = scanner.run_audit()

    assert audit.summary.trade_tick_detected_mismatch is False
    assert audit.summary.converter_trade_tick_instrument_count is None


def test_converter_report_sibling_discovery(tmp_path: Path) -> None:
    """Viewer should discover catalog_root.parent/convert_reports/YYYY-MM-DD.json."""
    root = tmp_path / "nautilus_data" / "catalog"
    iid = "BTCUSDT.BINANCE"
    _write_trade_parquet(root / "data" / "trade_tick" / iid / "part-0.parquet", [_ts(0), _ts(1)])
    _write_delta_parquet(root / "data" / "order_book_deltas" / iid / "part-0.parquet", [_ts(0)])
    (root / "data" / "currency_pair" / iid).mkdir(parents=True, exist_ok=True)

    report_dir = root.parent / "convert_reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "2026-04-25.json"
    report_path.write_text(json.dumps({
        "date": "2026-04-25",
        "timestamp": "2026-04-25T12:00:00Z",
        "status": "ok",
        "catalog_root": str(root),
        "report_paths": [str(report_path)],
        "data_presence": {"instruments_with_trades": 1, "no_data_list": []},
        "per_symbol_fenced_ranges": {},
    }), encoding="utf-8")

    audit = CatalogScanner(root, convert_report_date="2026-04-25").run_audit()

    assert audit.summary.convert_report_found is True
    assert audit.summary.convert_report_path == str(report_path)
    assert audit.summary.convert_report_date == "2026-04-25"
    assert audit.summary.convert_report_status == "ok"
    assert audit.summary.convert_report_matches_catalog_root is True


# ── Tests: 9. debug_trade_tick() ─────────────────────────────────────────────

def test_debug_trade_tick_spot_found(catalog_spot_with_trades: Path) -> None:
    """debug_trade_tick returns correct discovery info for a spot instrument."""
    scanner = CatalogScanner(catalog_spot_with_trades)
    debug = scanner.debug_trade_tick()

    assert debug["trade_tick_dir_exists"] is True
    assert debug["instrument_count"] == 1
    assert debug["total_parquet_files"] == 1
    assert debug["sample_schema"] is not None
    assert "ts_event" in debug["sample_schema"]
    assert len(debug["spot_instruments_found"]) == 1
    assert len(debug["futures_instruments_found"]) == 0
    first = debug["instruments"][0]
    assert first["instrument_id"] == "BTCUSDT.BINANCE"
    assert first["row_count"] == 5
    assert first["ts_event_min_ns"] == _ts(0)
    assert first["ts_event_max_ns"] == _ts(4)
    assert first["error"] is None


def test_debug_trade_tick_perp_found(catalog_perp_with_trades: Path) -> None:
    """debug_trade_tick correctly classifies a perpetual instrument as futures."""
    scanner = CatalogScanner(catalog_perp_with_trades)
    debug = scanner.debug_trade_tick()

    assert debug["instrument_count"] == 1
    assert "BTCUSDT-PERP.BINANCE" in debug["futures_instruments_found"]
    assert len(debug["spot_instruments_found"]) == 0


def test_debug_trade_tick_empty_dir(catalog_perp_no_trades: Path) -> None:
    """debug_trade_tick on catalog with no trade_tick directory reports correctly."""
    scanner = CatalogScanner(catalog_perp_no_trades)
    debug = scanner.debug_trade_tick()

    # No trade_tick dir -> instrument_count 0
    assert debug["instrument_count"] == 0
    assert debug["total_parquet_files"] == 0


# ── Tests: 10. Schema tolerance ───────────────────────────────────────────────

def test_resolve_ts_for_parquet_event_ts_fallback(tmp_path: Path) -> None:
    """Scanner can resolve timestamp using 'event_ts' column when 'ts_event' absent."""
    from app.catalog_scan import CatalogScanner

    root = tmp_path / "catalog"
    iid = "ALTUSDT.BINANCE"
    parquet_path = root / "data" / "trade_tick" / iid / "part-0.parquet"
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    # Use 'event_ts' instead of 'ts_event'
    ts_values = [_ts(0), _ts(1), _ts(2)]
    table = pa.table({
        "event_ts": pa.array(ts_values, type=pa.int64()),
        "price": pa.array([b"\x00" * 16] * 3, type=pa.binary()),
    })
    pq.write_table(table, str(parquet_path))
    (root / "data" / "currency_pair" / iid).mkdir(parents=True, exist_ok=True)

    scanner = CatalogScanner(root)
    audit = scanner.run_audit()
    inst = next(i for i in audit.instruments if i.instrument_id == iid)
    tt = inst.data_types["trade_tick"]

    # Should be present=True and stats successfully read
    assert tt.present is True
    assert tt.row_count_estimate == 3
    assert tt.ts_event_min_ns == _ts(0)
    assert tt.ts_event_max_ns == _ts(2)
    assert tt.corrupt is False


def test_audit_summary_trade_tick_viewer_count(catalog_mixed: Path) -> None:
    """audit.summary.viewer_trade_tick_instrument_count counts only present=True instruments."""
    scanner = CatalogScanner(catalog_mixed)
    audit = scanner.run_audit()

    # Only SOLUSDT.BINANCE has trades, SOLUSDT-PERP.BINANCE does not
    assert audit.summary.viewer_trade_tick_instrument_count == 1
    assert audit.summary.data_type_coverage["trade_tick"] == 1


def test_metadata_rows_without_timestamp_stats(tmp_path: Path) -> None:
    """metadata.num_rows must be counted even when row-group timestamp stats are absent."""
    root = tmp_path / "catalog"
    iid = "STATLESS.BINANCE"
    parquet_path = root / "data" / "trade_tick" / iid / "part-0.parquet"
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.table({
            "ts_event": pa.array([_ts(0), _ts(1), _ts(2)], type=pa.int64()),
            "price": pa.array([b"\x00" * 16] * 3, type=pa.binary()),
        }),
        str(parquet_path),
        write_statistics=False,
    )
    (root / "data" / "currency_pair" / iid).mkdir(parents=True, exist_ok=True)

    audit = CatalogScanner(root).run_audit()
    stat = next(i for i in audit.instruments if i.instrument_id == iid).data_types["trade_tick"]

    assert stat.present is True
    assert stat.status == "present_with_rows"
    assert stat.row_count_estimate == 3
    assert stat.timestamp_status == "fallback_read"
    assert stat.ts_event_min_ns == _ts(0)


def test_files_with_unsupported_timestamp_column_are_not_absent(tmp_path: Path) -> None:
    """Files with rows but no known timestamp column keep row counts and explicit timestamp status."""
    root = tmp_path / "catalog"
    iid = "NOTS.BINANCE"
    parquet_path = root / "data" / "trade_tick" / iid / "part-0.parquet"
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table({"sequence": pa.array([1, 2, 3], type=pa.int64())}), str(parquet_path))
    (root / "data" / "currency_pair" / iid).mkdir(parents=True, exist_ok=True)

    audit = CatalogScanner(root).run_audit()
    stat = next(i for i in audit.instruments if i.instrument_id == iid).data_types["trade_tick"]

    assert stat.present is True
    assert stat.status == "present_with_rows"
    assert stat.row_count_estimate == 3
    assert stat.timestamp_status == "missing_timestamp_column"
    assert stat.ts_event_min_ns is None


def test_file_count_with_scan_failure_becomes_present_unreadable(tmp_path: Path) -> None:
    """A parquet-looking file that cannot be read is explicit unreadable, not clean empty."""
    root = tmp_path / "catalog"
    iid = "BROKEN.BINANCE"
    parquet_path = root / "data" / "trade_tick" / iid / "part-0.parquet"
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    parquet_path.write_bytes(b"not a parquet file")
    (root / "data" / "currency_pair" / iid).mkdir(parents=True, exist_ok=True)

    audit = CatalogScanner(root).run_audit()
    stat = next(i for i in audit.instruments if i.instrument_id == iid).data_types["trade_tick"]

    assert stat.present is True
    assert stat.file_count == 1
    assert stat.status == "present_unreadable"
    assert stat.row_count_estimate == 0
    assert stat.error is not None


def test_order_book_deltas_included_in_readiness(catalog_perp_no_trades: Path) -> None:
    """Deltas-only instruments should be L2-ready instead of not-ready."""
    audit = CatalogScanner(catalog_perp_no_trades).run_audit()
    inst = next(i for i in audit.instruments if i.instrument_id == "ETHUSDT-PERP.BINANCE")

    assert inst.readiness.has_order_book_deltas is True
    assert inst.readiness.backtest_readiness_score == 70.0
    assert inst.readiness.readiness_status == "l2_ready"


def test_convert_report_overrides_audit_uncertainty_with_warning(tmp_path: Path) -> None:
    """Convert rows can classify readiness while audit mismatch lowers confidence."""
    root = tmp_path / "nautilus_data" / "catalog"
    iid = "OVERRIDE.BINANCE"
    for data_type in ("trade_tick", "order_book_deltas"):
        parquet_path = root / "data" / data_type / iid / "part-0.parquet"
        parquet_path.parent.mkdir(parents=True, exist_ok=True)
        parquet_path.write_bytes(b"not a parquet file")
    (root / "data" / "currency_pair" / iid).mkdir(parents=True, exist_ok=True)
    report_dir = root.parent / "convert_reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "2026-04-25.json").write_text(json.dumps({
        "date": "2026-04-25",
        "status": "ok",
        "catalog_root": str(root),
        "per_symbol_trade": {iid: {"ticks_written": 10}},
        "per_symbol_depth": {iid: {"deltas_written": 20}},
    }), encoding="utf-8")

    audit = CatalogScanner(root, convert_report_date="2026-04-25").run_audit()
    inst = next(i for i in audit.instruments if i.instrument_id == iid)

    assert inst.data_types["trade_tick"].status == "present_unreadable"
    assert inst.data_types["trade_tick"].row_count_source == "convert_report"
    assert inst.data_types["order_book_deltas"].row_count_estimate == 20
    assert inst.readiness.backtest_readiness_score == 100.0
    assert inst.audit_confidence_score < 100.0
    assert any(w.code == "convert_report_audit_mismatch" for w in audit.warnings)


def test_zero_rows_without_error_only_when_metadata_confirms_empty(tmp_path: Path) -> None:
    """No present file may report zero rows and no error unless parquet metadata confirms empty."""
    root = tmp_path / "catalog"
    iid = "EMPTY.BINANCE"
    parquet_path = root / "data" / "trade_tick" / iid / "part-0.parquet"
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table({"ts_event": pa.array([], type=pa.int64())}), str(parquet_path))
    (root / "data" / "currency_pair" / iid).mkdir(parents=True, exist_ok=True)

    audit = CatalogScanner(root).run_audit()
    stat = next(i for i in audit.instruments if i.instrument_id == iid).data_types["trade_tick"]

    assert stat.present is True
    assert stat.file_count == 1
    assert stat.row_count_estimate == 0
    assert stat.status == "present_empty"
    assert stat.error is None
