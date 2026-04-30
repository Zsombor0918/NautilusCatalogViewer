"""Tests for the updated readiness model and depth10 parser fixes.

Covers:
- TradeTick only → trade_only, is_backtest_ready False
- OrderBookDeltas only → l2_replay_ready, is_backtest_ready False
- TradeTick + OrderBookDeltas → full_ready, is_backtest_ready True
- OrderBookDepth10 only → depth10_inspection_only, is_backtest_ready False
- depth10 rows > 0 but no flat bid/ask columns → debug_depth10 returns parser_error
- Converter readiness_classification grouped-list format parsing
- Zero-padded depth10 levels (price=0,size=0) do NOT trigger zero_qty
- price>0 size=0 level DOES trigger zero_qty
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

from app.catalog_scan import CatalogScanner, _converter_readiness_classification
from app.l2_checks import ParsedL2Snapshot, _evaluate_levels, parsed_snapshot_from_record
from app.query import CatalogQueryService
from app.scoring import compute_readiness_breakdown, readiness_status_for_presence


# ── Helpers ──────────────────────────────────────────────────────────────────

def _write_parquet(path: Path, ts_values: list[int], extra_columns: dict[str, list[Any]] | None = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns: dict[str, pa.Array] = {"ts_event": pa.array(ts_values, type=pa.int64())}
    if extra_columns:
        for name, values in extra_columns.items():
            if values and isinstance(values[0], bytes):
                columns[name] = pa.array(values, type=pa.binary())
            elif values and isinstance(values[0], float):
                columns[name] = pa.array(values, type=pa.float64())
            else:
                columns[name] = pa.array(values, type=pa.int64())
    pq.write_table(pa.table(columns), str(path))
    return path


def _fixed_decimal_bytes(value: float, precision: int = 16) -> bytes:
    """Encode a float as Nautilus fixed-point int64 bytes (little-endian)."""
    raw = int(round(value * 10**precision))
    return raw.to_bytes(8, byteorder="little", signed=True)


def _make_depth10_parquet(path: Path, ts_values: list[int], best_bid: float, best_ask: float) -> Path:
    """Write a flat-format depth10 parquet with valid bid/ask levels."""
    extra: dict[str, list] = {}
    for level in range(10):
        bid_price = best_bid - level * 0.01
        bid_size = 1.0 + level * 0.1
        ask_price = best_ask + level * 0.01
        ask_size = 1.0 + level * 0.1
        extra[f"bid_price_{level}"] = [_fixed_decimal_bytes(bid_price)] * len(ts_values)
        extra[f"ask_price_{level}"] = [_fixed_decimal_bytes(ask_price)] * len(ts_values)
        extra[f"bid_size_{level}"] = [_fixed_decimal_bytes(bid_size)] * len(ts_values)
        extra[f"ask_size_{level}"] = [_fixed_decimal_bytes(ask_size)] * len(ts_values)
    extra["flags"] = [0] * len(ts_values)
    extra["sequence"] = list(range(len(ts_values)))
    return _write_parquet(path, ts_values, extra)


# ── Scoring tests ─────────────────────────────────────────────────────────────

class TestReadinessStatusForPresence:
    def test_full_ready(self):
        assert readiness_status_for_presence(
            has_trade_rows=True, has_delta_rows=True, has_depth_rows=False, partial_unreadable=False
        ) == "full_ready"

    def test_full_ready_with_depth10(self):
        assert readiness_status_for_presence(
            has_trade_rows=True, has_delta_rows=True, has_depth_rows=True, partial_unreadable=False
        ) == "full_ready"

    def test_l2_replay_ready(self):
        assert readiness_status_for_presence(
            has_trade_rows=False, has_delta_rows=True, has_depth_rows=False, partial_unreadable=False
        ) == "l2_replay_ready"

    def test_trade_only(self):
        assert readiness_status_for_presence(
            has_trade_rows=True, has_delta_rows=False, has_depth_rows=False, partial_unreadable=False
        ) == "trade_only"

    def test_depth10_inspection_only(self):
        assert readiness_status_for_presence(
            has_trade_rows=False, has_delta_rows=False, has_depth_rows=True, partial_unreadable=False
        ) == "depth10_inspection_only"

    def test_partial_unreadable_overrides(self):
        assert readiness_status_for_presence(
            has_trade_rows=True, has_delta_rows=True, has_depth_rows=True, partial_unreadable=True
        ) == "partial_unreadable"

    def test_not_ready(self):
        assert readiness_status_for_presence(
            has_trade_rows=False, has_delta_rows=False, has_depth_rows=False, partial_unreadable=False
        ) == "not_ready"


class TestComputeReadinessBreakdown:
    def test_trade_only_score_60(self):
        score, _ = compute_readiness_breakdown(
            has_trade_tick=True, has_order_book_deltas=False, has_order_book_depths=False,
            trade_row_count=100, delta_row_count=0, depth_row_count=0,
        )
        assert score == 60.0

    def test_deltas_only_score_70(self):
        score, _ = compute_readiness_breakdown(
            has_trade_tick=False, has_order_book_deltas=True, has_order_book_depths=False,
            trade_row_count=0, delta_row_count=100, depth_row_count=0,
        )
        assert score == 70.0

    def test_depth10_only_score_50(self):
        score, _ = compute_readiness_breakdown(
            has_trade_tick=False, has_order_book_deltas=False, has_order_book_depths=True,
            trade_row_count=0, delta_row_count=0, depth_row_count=100,
        )
        assert score == 50.0

    def test_full_ready_score_100(self):
        score, _ = compute_readiness_breakdown(
            has_trade_tick=True, has_order_book_deltas=True, has_order_book_depths=False,
            trade_row_count=100, delta_row_count=100, depth_row_count=0,
        )
        assert score == 100.0

    def test_depth10_does_not_push_to_full_ready(self):
        # depth10 alone + trade (no deltas) must NOT be full_ready
        score, _ = compute_readiness_breakdown(
            has_trade_tick=True, has_order_book_deltas=False, has_order_book_depths=True,
            trade_row_count=100, delta_row_count=0, depth_row_count=100,
        )
        assert score < 100.0


# ── Catalog scanner integration tests ────────────────────────────────────────

class TestScannerReadinessStatuses:
    def test_trade_only(self, tmp_path: Path):
        root = tmp_path / "catalog"
        inst = "BTCUSDT-PERP.BINANCE"
        _write_parquet(
            root / "data" / "trade_tick" / inst / "part-0.parquet",
            ts_values=[1_000_000_000, 2_000_000_000],
            extra_columns={"price": [b"\x00" * 8] * 2, "size": [b"\x00" * 8] * 2, "aggressor_side": [1, 2]},
        )
        (root / "data" / "crypto_perpetual" / inst).mkdir(parents=True, exist_ok=True)
        scanner = CatalogScanner(catalog_root=root)
        audit = scanner.run_audit()
        item = next(i for i in audit.instruments if i.instrument_id == inst)
        assert item.readiness.is_backtest_ready is False
        assert item.readiness.readiness_status == "trade_only"

    def test_deltas_only_l2_replay_ready(self, tmp_path: Path):
        root = tmp_path / "catalog"
        inst = "BTCUSDT-PERP.BINANCE"
        _write_parquet(
            root / "data" / "order_book_deltas" / inst / "part-0.parquet",
            ts_values=[1_000_000_000, 2_000_000_000],
            extra_columns={"action": [1, 1], "side": [1, 2], "price": [b"\x00" * 8] * 2, "size": [b"\x00" * 8] * 2, "flags": [0, 0], "sequence": [1, 2]},
        )
        (root / "data" / "crypto_perpetual" / inst).mkdir(parents=True, exist_ok=True)
        scanner = CatalogScanner(catalog_root=root)
        audit = scanner.run_audit()
        item = next(i for i in audit.instruments if i.instrument_id == inst)
        assert item.readiness.is_backtest_ready is False
        assert item.readiness.readiness_status == "l2_replay_ready"

    def test_full_ready(self, tmp_path: Path):
        root = tmp_path / "catalog"
        inst = "BTCUSDT-PERP.BINANCE"
        _write_parquet(
            root / "data" / "trade_tick" / inst / "part-0.parquet",
            ts_values=[1_000_000_000, 2_000_000_000],
            extra_columns={"price": [b"\x00" * 8] * 2, "size": [b"\x00" * 8] * 2, "aggressor_side": [1, 2]},
        )
        _write_parquet(
            root / "data" / "order_book_deltas" / inst / "part-0.parquet",
            ts_values=[1_000_000_000, 2_000_000_000],
            extra_columns={"action": [1, 1], "side": [1, 2], "price": [b"\x00" * 8] * 2, "size": [b"\x00" * 8] * 2, "flags": [0, 0], "sequence": [1, 2]},
        )
        (root / "data" / "crypto_perpetual" / inst).mkdir(parents=True, exist_ok=True)
        scanner = CatalogScanner(catalog_root=root)
        audit = scanner.run_audit()
        item = next(i for i in audit.instruments if i.instrument_id == inst)
        assert item.readiness.is_backtest_ready is True
        assert item.readiness.readiness_status == "full_ready"

    def test_depth10_only_not_backtest_ready(self, tmp_path: Path):
        root = tmp_path / "catalog"
        inst = "ZECUSDT.BINANCE"
        _make_depth10_parquet(
            root / "data" / "order_book_depths" / inst / "part-0.parquet",
            ts_values=[1_000_000_000, 2_000_000_000, 3_000_000_000],
            best_bid=50.0,
            best_ask=50.05,
        )
        (root / "data" / "currency_pair" / inst).mkdir(parents=True, exist_ok=True)
        scanner = CatalogScanner(catalog_root=root)
        audit = scanner.run_audit()
        item = next(i for i in audit.instruments if i.instrument_id == inst)
        assert item.readiness.is_backtest_ready is False
        assert item.readiness.readiness_status == "depth10_inspection_only"
        assert item.readiness.depth10_inspection_ready is True


# ── Depth10 parser / debug tests ─────────────────────────────────────────────

class TestDepth10Parser:
    def test_debug_no_files(self, tmp_path: Path):
        root = tmp_path / "catalog"
        qs = CatalogQueryService(root)
        debug = qs.debug_depth10("NODATA.BINANCE")
        assert debug.parser_ok is False
        assert debug.parser_error is not None
        assert "No order_book_depths" in debug.parser_error

    def test_debug_wrong_schema(self, tmp_path: Path):
        """Depth10 parquet file with only ts_event (no bid/ask columns)."""
        root = tmp_path / "catalog"
        inst = "ZECUSDT.BINANCE"
        _write_parquet(
            root / "data" / "order_book_depths" / inst / "part-0.parquet",
            ts_values=[1_000_000_000, 2_000_000_000],
        )
        qs = CatalogQueryService(root)
        debug = qs.debug_depth10(inst)
        assert debug.parser_ok is False
        assert debug.parser_error is not None
        assert "no bid/ask level columns" in debug.parser_error.lower() or "Missing" in debug.parser_error

    def test_debug_correct_schema(self, tmp_path: Path):
        """Depth10 parquet file with flat bid/ask columns decodes correctly."""
        root = tmp_path / "catalog"
        inst = "ZECUSDT.BINANCE"
        _make_depth10_parquet(
            root / "data" / "order_book_depths" / inst / "part-0.parquet",
            ts_values=[1_000_000_000, 2_000_000_000],
            best_bid=50.0,
            best_ask=50.05,
        )
        qs = CatalogQueryService(root)
        debug = qs.debug_depth10(inst)
        assert debug.parser_ok is True
        assert debug.parser_error is None
        assert len(debug.depth_cols_found) == 40  # 4 prefixes × 10 levels
        assert debug.row_count == 2

    def test_l2_timeseries_returns_error_for_missing_schema(self, tmp_path: Path):
        """l2_timeseries endpoint returns error (not empty) when schema is wrong."""
        root = tmp_path / "catalog"
        inst = "ZECUSDT.BINANCE"
        _write_parquet(
            root / "data" / "order_book_depths" / inst / "part-0.parquet",
            ts_values=[1_000_000_000, 2_000_000_000],
        )
        qs = CatalogQueryService(root)
        result = qs.get_l2_timeseries(inst)
        assert result.error is not None
        assert "no bid/ask" in result.error.lower() or "parser" in result.error.lower() or "schema" in result.error.lower() or "missing" in result.error.lower()

    def test_valid_depth10_timeseries_has_prices(self, tmp_path: Path):
        """Valid depth10 data returns non-None best_bid and best_ask."""
        root = tmp_path / "catalog"
        inst = "ZECUSDT.BINANCE"
        _make_depth10_parquet(
            root / "data" / "order_book_depths" / inst / "part-0.parquet",
            ts_values=[1_000_000_000, 2_000_000_000, 3_000_000_000],
            best_bid=50.0,
            best_ask=50.05,
        )
        qs = CatalogQueryService(root)
        result = qs.get_l2_timeseries(inst, mode="raw")
        assert result.error is None
        assert result.total_rows == 3
        assert result.points[0].best_bid is not None
        assert result.points[0].best_ask is not None
        assert result.points[0].best_ask > result.points[0].best_bid


# ── L2 checks zero_qty false positive tests ──────────────────────────────────

class TestL2ZeroQtyFalsePositive:
    def test_padded_zeros_not_flagged(self):
        """Levels with price=0,size=0 (padding) must NOT trigger has_zero_qty."""
        bid_levels = [(50.0, 1.0), (49.99, 0.5)] + [(0.0, 0.0)] * 8
        ask_levels = [(50.05, 1.0), (50.10, 0.5)] + [(0.0, 0.0)] * 8
        snap = _evaluate_levels(index=0, ts_event_ns=1_000_000_000, bid_levels=bid_levels, ask_levels=ask_levels)
        assert snap.has_zero_qty is False

    def test_valid_price_zero_size_triggers_zero_qty(self):
        """A level with price > 0 but size == 0 MUST trigger has_zero_qty."""
        bid_levels = [(50.0, 0.0)] + [(49.99, 1.0)] * 9  # first level has size=0
        ask_levels = [(50.05, 1.0)] * 10
        snap = _evaluate_levels(index=0, ts_event_ns=1_000_000_000, bid_levels=bid_levels, ask_levels=ask_levels)
        assert snap.has_zero_qty is True

    def test_all_zeros_gives_empty_side_not_zero_qty(self):
        """All-zero levels (parsing failure fallback) give empty_side but NOT zero_qty."""
        bid_levels = [(0.0, 0.0)] * 10
        ask_levels = [(0.0, 0.0)] * 10
        snap = _evaluate_levels(index=0, ts_event_ns=1_000_000_000, bid_levels=bid_levels, ask_levels=ask_levels)
        assert snap.has_zero_qty is False
        assert snap.has_empty_side is True

    def test_from_record_zero_padded(self):
        """parsed_snapshot_from_record with zero-padded trailing levels works correctly."""
        # bid_price_0..1 valid, rest are 0
        record = {"ts_event": 1_000_000_000}
        FP = 10 ** 16

        def enc(v):
            return int(v * FP).to_bytes(8, byteorder="little", signed=True)

        record["bid_price_0"] = enc(50.0)
        record["bid_size_0"] = enc(1.0)
        record["bid_price_1"] = enc(49.99)
        record["bid_size_1"] = enc(0.5)
        for i in range(2, 10):
            record[f"bid_price_{i}"] = enc(0.0)
            record[f"bid_size_{i}"] = enc(0.0)
        record["ask_price_0"] = enc(50.05)
        record["ask_size_0"] = enc(1.0)
        for i in range(1, 10):
            record[f"ask_price_{i}"] = enc(50.05 + i * 0.01)
            record[f"ask_size_{i}"] = enc(0.5)

        snap = parsed_snapshot_from_record(record)
        assert snap.has_zero_qty is False
        assert snap.best_bid == pytest.approx(50.0, abs=1e-6)
        assert snap.best_ask == pytest.approx(50.05, abs=1e-6)
        assert snap.is_crossed is False


# ── Converter readiness_classification grouped-list parsing ──────────────────

class TestConverterReadinessClassification:
    def test_grouped_list_format(self):
        raw = {
            "readiness_classification": {
                "full_ready": ["ZECUSDT.BINANCE", "BTCUSDT-PERP.BINANCE"],
                "l2_ready": ["ETHUSDT.BINANCE"],
                "not_ready": ["XRPUSDT.BINANCE"],
            }
        }
        assert _converter_readiness_classification(raw, "ZECUSDT.BINANCE") == "full_ready"
        assert _converter_readiness_classification(raw, "BTCUSDT-PERP.BINANCE") == "full_ready"
        assert _converter_readiness_classification(raw, "ETHUSDT.BINANCE") == "l2_ready"
        assert _converter_readiness_classification(raw, "XRPUSDT.BINANCE") == "not_ready"
        assert _converter_readiness_classification(raw, "UNKNOWN.BINANCE") is None

    def test_grouped_list_with_exchange_key(self):
        raw = {
            "readiness_classification": {
                "full_ready": ["BINANCE_SPOT/ZECUSDT"],
                "not_ready": [],
            }
        }
        assert _converter_readiness_classification(raw, "ZECUSDT.BINANCE") == "full_ready"

    def test_direct_map_format(self):
        raw = {
            "readiness_classification": {
                "ZECUSDT.BINANCE": "full_ready",
                "ETHUSDT.BINANCE": "trade_only",
            }
        }
        assert _converter_readiness_classification(raw, "ZECUSDT.BINANCE") == "full_ready"
        assert _converter_readiness_classification(raw, "ETHUSDT.BINANCE") == "trade_only"

    def test_dict_value_format(self):
        raw = {
            "readiness_classification": {
                "ZECUSDT.BINANCE": {"status": "full_ready", "score": 100},
            }
        }
        assert _converter_readiness_classification(raw, "ZECUSDT.BINANCE") == "full_ready"

    def test_missing_key_returns_none(self):
        raw = {"readiness_classification": {"BTCUSDT.BINANCE": "full_ready"}}
        assert _converter_readiness_classification(raw, "ZECUSDT.BINANCE") is None

    def test_no_section_returns_none(self):
        assert _converter_readiness_classification({}, "ZECUSDT.BINANCE") is None
