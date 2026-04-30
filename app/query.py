from __future__ import annotations

import csv
import io
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import pyarrow.dataset as ds

from .cache import QueryCache, build_cache_key, compute_files_signature
from .scoring import compute_readiness_breakdown, compute_readiness_score, readiness_status_for_score
from .l2_checks import (
    DEPTH_LEVELS,
    compute_gap_entries,
    decode_fixed_decimal,
    estimate_missing_ratio,
    ns_to_iso,
    parsed_snapshot_from_record,
    parsed_to_snapshot,
    parsed_to_summary,
    quality_from_snapshots,
)
from .models import (
    CoverageResponse,
    CoverageSummary,
    DeltaSeriesPoint,
    DeltasResponse,
    DeltasSummaryResponse,
    InstrumentSearchItem,
    InstrumentSearchResponse,
    L2QualityResponse,
    L2SnapshotResponse,
    L2TimeseriesPoint,
    L2TimeseriesResponse,
    ReadinessResponse,
    ReadinessResult,
    TradeSeriesPoint,
    TradesResponse,
)


EVENT_DATA_TYPES: tuple[str, ...] = ("trade_tick", "order_book_deltas", "order_book_depths")
TRADE_SIDE_MAP = {1: "BUYER", 2: "SELLER"}
DELTA_ACTION_MAP = {1: "ADD", 2: "UPDATE", 3: "DELETE", 4: "CLEAR"}
DELTA_SIDE_MAP = {1: "BID", 2: "ASK"}


def utc_now_iso() -> str:
    return datetime.now(tz=timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def instrument_type_from_id(instrument_id: str) -> str:
    return "crypto_perpetual" if "-PERP." in instrument_id else "currency_pair"


def parse_time_value(value: str | int | None) -> int | None:
    if value is None or value == "":
        return None
    if isinstance(value, int):
        return value
    text = str(value).strip()
    if text.isdigit():
        return int(text)

    normalized = text.replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise ValueError(f"\u00c9rv\u00e9nytelen id\u0151form\u00e1tum: {value}") from exc

    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    dt_utc = dt.astimezone(timezone.utc)
    return int(dt_utc.timestamp() * 1_000_000_000)


def _time_filter(ts_field: str, from_ns: int | None, to_ns: int | None) -> ds.Expression | None:
    expression: ds.Expression | None = None
    if from_ns is not None:
        expression = ds.field(ts_field) >= from_ns
    if to_ns is not None:
        end_expression = ds.field(ts_field) <= to_ns
        expression = end_expression if expression is None else expression & end_expression
    return expression


def _evenly_spaced_indices(length: int, max_points: int) -> np.ndarray:
    if length <= max_points:
        return np.arange(length, dtype=np.int64)
    return np.unique(np.linspace(0, length - 1, num=max_points, dtype=np.int64))


def _sort_table_by_ts(table_dict: dict[str, list[Any]], ts_field: str) -> np.ndarray:
    ts_values = np.asarray(table_dict.get(ts_field, []), dtype=np.int64)
    if ts_values.size == 0:
        return np.asarray([], dtype=np.int64)
    return np.argsort(ts_values, kind="stable")


class CatalogQueryService:
    def __init__(self, catalog_root: Path | str, cache_dir: Path | str | None = None) -> None:
        self.catalog_root = Path(catalog_root).expanduser().resolve()
        project_root = Path(__file__).resolve().parent.parent
        self.cache = QueryCache(cache_dir or (project_root / "state" / "api_cache"))
        self._l2_index_cache: dict[str, tuple[str, np.ndarray]] = {}

    @property
    def data_root(self) -> Path:
        return self.catalog_root / "data"

    def _instrument_dir(self, data_type: str, instrument_id: str) -> Path:
        return self.data_root / data_type / instrument_id

    def list_files(self, data_type: str, instrument_id: str) -> list[Path]:
        directory = self._instrument_dir(data_type, instrument_id)
        if not directory.exists():
            return []
        return sorted(path for path in directory.glob("*.parquet") if path.is_file())

    def _signature_for(self, *file_lists: list[Path]) -> str:
        files: list[Path] = []
        for item in file_lists:
            files.extend(item)
        return compute_files_signature(files)

    def _dataset(self, data_type: str, instrument_id: str) -> ds.Dataset | None:
        files = self.list_files(data_type, instrument_id)
        if not files:
            return None
        return ds.dataset([str(path) for path in files], format="parquet")

    def _resolve_ts_field(self, schema_names: list[str]) -> str:
        candidates = ["ts_event", "event_ts", "ts"]
        for candidate in candidates:
            if candidate in schema_names:
                return candidate
        for name in schema_names:
            if "ts_event" in name:
                return name
        raise KeyError("Nem tal\u00e1lhat\u00f3 ts_event mez\u0151 a s\u00e9m\u00e1ban.")

    def _resolve_depth_columns(self, schema_names: list[str]) -> list[str]:
        columns: list[str] = []
        for prefix in ("bid_price_", "ask_price_", "bid_size_", "ask_size_"):
            for level in range(DEPTH_LEVELS):
                name = f"{prefix}{level}"
                if name in schema_names:
                    columns.append(name)
        for optional_name in ("sequence", "flags"):
            if optional_name in schema_names:
                columns.append(optional_name)
        return columns

    # ── Instrument search ───────────────────────────────────────────────────

    def get_instruments(self, inventory_items: list[dict[str, Any]], type_filter: str | None = None, q: str | None = None) -> InstrumentSearchResponse:
        filtered = []
        search = (q or "").strip().lower()
        for item in inventory_items:
            instrument_type = item["instrument_type"]
            if type_filter and instrument_type != type_filter:
                continue
            instrument_id = item["instrument_id"]
            if search and search not in instrument_id.lower():
                continue
            filtered.append(
                InstrumentSearchItem(
                    instrument_id=instrument_id,
                    instrument_type=instrument_type,
                    has_trade_tick=bool(item["coverage"].get("trade_tick", {}).get("present", False)),
                    has_order_book_deltas=bool(item["coverage"].get("order_book_deltas", {}).get("present", False)),
                    has_order_book_depths=bool(item["coverage"].get("order_book_depths", {}).get("present", False)),
                ),
            )

        return InstrumentSearchResponse(
            catalog_root=str(self.catalog_root),
            total=len(filtered),
            items=filtered,
        )

    # ── Coverage ────────────────────────────────────────────────────────────

    def _coverage_for_type(
        self,
        *,
        data_type: str,
        instrument_id: str,
        from_ns: int | None,
        to_ns: int | None,
    ) -> CoverageSummary:
        files = self.list_files(data_type, instrument_id)
        if not files:
            return CoverageSummary(data_type=data_type, present=False)

        try:
            dataset = ds.dataset([str(path) for path in files], format="parquet")
            ts_field = self._resolve_ts_field(dataset.schema.names)
            filter_expr = _time_filter(ts_field, from_ns, to_ns)
            row_count = dataset.count_rows(filter=filter_expr)
            ts_table = dataset.to_table(columns=[ts_field], filter=filter_expr)
        except Exception as exc:
            return CoverageSummary(
                data_type=data_type,
                present=True,
                file_count=len(files),
                error=str(exc),
            )

        if ts_table.num_rows == 0:
            return CoverageSummary(
                data_type=data_type,
                present=True,
                file_count=len(files),
                row_count=0,
            )

        table_dict = ts_table.to_pydict()
        order = _sort_table_by_ts(table_dict, ts_field)
        ts_values = np.asarray(table_dict[ts_field], dtype=np.int64)[order]
        gaps = compute_gap_entries(ts_values.tolist(), top_n=10)
        duration_seconds = float(ts_values[-1] - ts_values[0]) / 1_000_000_000 if len(ts_values) > 1 else 0.0
        threshold_ns = 300 * 1_000_000_000
        session_break_count = int(np.sum(np.diff(ts_values) >= threshold_ns)) if len(ts_values) > 1 else 0

        return CoverageSummary(
            data_type=data_type,
            present=True,
            file_count=len(files),
            row_count=int(row_count),
            ts_event_min_ns=int(ts_values[0]),
            ts_event_min_iso=ns_to_iso(int(ts_values[0])),
            ts_event_max_ns=int(ts_values[-1]),
            ts_event_max_iso=ns_to_iso(int(ts_values[-1])),
            duration_seconds=duration_seconds,
            max_gap_ns=gaps[0].gap_ns if gaps else None,
            max_gap_seconds=gaps[0].gap_seconds if gaps else None,
            missing_ratio_estimate=estimate_missing_ratio(ts_values.tolist()),
            session_break_count=session_break_count,
            top_gaps=gaps,
        )

    def get_coverage(self, instrument_id: str, from_value: str | int | None = None, to_value: str | int | None = None) -> CoverageResponse:
        from_ns = parse_time_value(from_value)
        to_ns = parse_time_value(to_value)
        trade_files = self.list_files("trade_tick", instrument_id)
        delta_files = self.list_files("order_book_deltas", instrument_id)
        depth_files = self.list_files("order_book_depths", instrument_id)
        signature = self._signature_for(trade_files, delta_files, depth_files)
        key = build_cache_key("coverage", instrument_id=instrument_id, from_ns=from_ns, to_ns=to_ns)
        cached = self.cache.get_model(key, signature, CoverageResponse)
        if cached is not None:
            return cached

        response = CoverageResponse(
            instrument_id=instrument_id,
            instrument_type=instrument_type_from_id(instrument_id),
            from_ns=from_ns,
            from_iso=ns_to_iso(from_ns),
            to_ns=to_ns,
            to_iso=ns_to_iso(to_ns),
            generated_at=utc_now_iso(),
            coverage={
                "trade_tick": self._coverage_for_type(
                    data_type="trade_tick",
                    instrument_id=instrument_id,
                    from_ns=from_ns,
                    to_ns=to_ns,
                ),
                "order_book_deltas": self._coverage_for_type(
                    data_type="order_book_deltas",
                    instrument_id=instrument_id,
                    from_ns=from_ns,
                    to_ns=to_ns,
                ),
                "order_book_depths": self._coverage_for_type(
                    data_type="order_book_depths",
                    instrument_id=instrument_id,
                    from_ns=from_ns,
                    to_ns=to_ns,
                ),
            },
        )
        self.cache.set_model(key, signature, response)
        return response

    # ── Trades ──────────────────────────────────────────────────────────────

    def _trade_dataframe(self, instrument_id: str, from_ns: int | None, to_ns: int | None) -> tuple[pd.DataFrame, str | None]:
        dataset = self._dataset("trade_tick", instrument_id)
        if dataset is None:
            return pd.DataFrame(), None

        try:
            ts_field = self._resolve_ts_field(dataset.schema.names)
            columns = [ts_field]
            for name in ("price", "size", "aggressor_side", "trade_id"):
                if name in dataset.schema.names:
                    columns.append(name)
            table = dataset.to_table(columns=columns, filter=_time_filter(ts_field, from_ns, to_ns))
        except Exception as exc:
            return pd.DataFrame(), str(exc)

        if table.num_rows == 0:
            return pd.DataFrame(), None

        table_dict = table.to_pydict()
        order = _sort_table_by_ts(table_dict, ts_field)
        row_count = len(table_dict[ts_field])
        df = pd.DataFrame(
            {
                "ts_event_ns": np.asarray(table_dict[ts_field], dtype=np.int64)[order],
                "price": np.asarray([decode_fixed_decimal(value) for value in table_dict.get("price", [])], dtype=float)[order],
                "size": np.asarray([decode_fixed_decimal(value) for value in table_dict.get("size", [])], dtype=float)[order],
                "aggressor_side": np.asarray(
                    [TRADE_SIDE_MAP.get(int(value), "UNKNOWN") for value in table_dict.get("aggressor_side", [0] * row_count)],
                    dtype=object,
                )[order],
                "trade_id": np.asarray(table_dict.get("trade_id", [None] * row_count), dtype=object)[order],
            },
        )
        return df, None

    def get_trades(
        self,
        instrument_id: str,
        *,
        from_value: str | int | None = None,
        to_value: str | int | None = None,
        mode: Literal["raw", "agg"] = "raw",
        bucket_s: int = 60,
        max_points: int = 10_000,
    ) -> TradesResponse:
        from_ns = parse_time_value(from_value)
        to_ns = parse_time_value(to_value)
        files = self.list_files("trade_tick", instrument_id)
        signature = self._signature_for(files)
        key = build_cache_key(
            "trades",
            instrument_id=instrument_id,
            from_ns=from_ns,
            to_ns=to_ns,
            mode=mode,
            bucket_s=bucket_s,
            max_points=max_points,
        )
        cached = self.cache.get_model(key, signature, TradesResponse)
        if cached is not None:
            return cached

        df, error = self._trade_dataframe(instrument_id, from_ns, to_ns)
        if error is not None:
            return TradesResponse(
                instrument_id=instrument_id,
                instrument_type=instrument_type_from_id(instrument_id),
                mode=mode,
                bucket_s=bucket_s,
                max_points=max_points,
                generated_at=utc_now_iso(),
                error=error,
            )

        if df.empty:
            response = TradesResponse(
                instrument_id=instrument_id,
                instrument_type=instrument_type_from_id(instrument_id),
                mode=mode,
                bucket_s=bucket_s,
                max_points=max_points,
                total_rows=0,
                returned_points=0,
                from_ns=from_ns,
                from_iso=ns_to_iso(from_ns),
                to_ns=to_ns,
                to_iso=ns_to_iso(to_ns),
                generated_at=utc_now_iso(),
            )
            self.cache.set_model(key, signature, response)
            return response

        points: list[TradeSeriesPoint] = []
        if mode == "agg":
            bucket_ns = max(1, bucket_s) * 1_000_000_000
            working = df.copy()
            working["bucket_ns"] = (working["ts_event_ns"] // bucket_ns) * bucket_ns
            grouped = working.groupby("bucket_ns", as_index=False).agg(
                avg_price=("price", "mean"),
                last_price=("price", "last"),
                min_price=("price", "min"),
                max_price=("price", "max"),
                volume=("size", "sum"),
                trade_count=("size", "count"),
            )
            order = _evenly_spaced_indices(len(grouped), max_points)
            for row in grouped.iloc[order].itertuples(index=False):
                ts_event_ns = int(row.bucket_ns)
                points.append(
                    TradeSeriesPoint(
                        ts_event_ns=ts_event_ns,
                        ts_event_iso=ns_to_iso(ts_event_ns) or "",
                        trade_count=int(row.trade_count),
                        volume=float(row.volume),
                        avg_price=float(row.avg_price),
                        last_price=float(row.last_price),
                        min_price=float(row.min_price),
                        max_price=float(row.max_price),
                    ),
                )
        else:
            order = _evenly_spaced_indices(len(df), max_points)
            for row in df.iloc[order].itertuples(index=False):
                points.append(
                    TradeSeriesPoint(
                        ts_event_ns=int(row.ts_event_ns),
                        ts_event_iso=ns_to_iso(int(row.ts_event_ns)) or "",
                        price=float(row.price),
                        size=float(row.size),
                        aggressor_side=str(row.aggressor_side),
                        trade_id=str(row.trade_id) if row.trade_id is not None else None,
                    ),
                )

        response = TradesResponse(
            instrument_id=instrument_id,
            instrument_type=instrument_type_from_id(instrument_id),
            mode=mode,
            bucket_s=bucket_s,
            max_points=max_points,
            total_rows=len(df),
            returned_points=len(points),
            from_ns=int(df["ts_event_ns"].iloc[0]),
            from_iso=ns_to_iso(int(df["ts_event_ns"].iloc[0])),
            to_ns=int(df["ts_event_ns"].iloc[-1]),
            to_iso=ns_to_iso(int(df["ts_event_ns"].iloc[-1])),
            generated_at=utc_now_iso(),
            points=points,
        )
        self.cache.set_model(key, signature, response)
        return response

    # ── Deltas (primary order book data) ────────────────────────────────────

    def _delta_dataframe(self, instrument_id: str, from_ns: int | None, to_ns: int | None) -> tuple[pd.DataFrame, str | None]:
        dataset = self._dataset("order_book_deltas", instrument_id)
        if dataset is None:
            return pd.DataFrame(), None

        try:
            ts_field = self._resolve_ts_field(dataset.schema.names)
            columns = [ts_field]
            for name in ("action", "side", "price", "size", "flags", "sequence"):
                if name in dataset.schema.names:
                    columns.append(name)
            table = dataset.to_table(columns=columns, filter=_time_filter(ts_field, from_ns, to_ns))
        except Exception as exc:
            return pd.DataFrame(), str(exc)

        if table.num_rows == 0:
            return pd.DataFrame(), None

        table_dict = table.to_pydict()
        order = _sort_table_by_ts(table_dict, ts_field)
        row_count = len(table_dict[ts_field])

        df_dict: dict[str, Any] = {
            "ts_event_ns": np.asarray(table_dict[ts_field], dtype=np.int64)[order],
        }
        if "action" in table_dict:
            df_dict["action"] = np.asarray(
                [DELTA_ACTION_MAP.get(int(v), "UNKNOWN") for v in table_dict["action"]],
                dtype=object,
            )[order]
        else:
            df_dict["action"] = np.asarray(["UNKNOWN"] * row_count, dtype=object)[order]

        if "side" in table_dict:
            df_dict["side"] = np.asarray(
                [DELTA_SIDE_MAP.get(int(v), "UNKNOWN") for v in table_dict["side"]],
                dtype=object,
            )[order]
        else:
            df_dict["side"] = np.asarray(["UNKNOWN"] * row_count, dtype=object)[order]

        if "price" in table_dict:
            df_dict["price"] = np.asarray(
                [decode_fixed_decimal(v) for v in table_dict["price"]], dtype=float,
            )[order]
        else:
            df_dict["price"] = np.full(row_count, np.nan, dtype=float)[order]

        if "size" in table_dict:
            df_dict["size"] = np.asarray(
                [decode_fixed_decimal(v) for v in table_dict["size"]], dtype=float,
            )[order]
        else:
            df_dict["size"] = np.full(row_count, np.nan, dtype=float)[order]

        if "flags" in table_dict:
            df_dict["flags"] = np.asarray(table_dict["flags"], dtype=np.int64)[order]
        else:
            df_dict["flags"] = np.zeros(row_count, dtype=np.int64)[order]

        if "sequence" in table_dict:
            df_dict["sequence"] = np.asarray(table_dict["sequence"], dtype=np.int64)[order]
        else:
            df_dict["sequence"] = np.zeros(row_count, dtype=np.int64)[order]

        df = pd.DataFrame(df_dict)
        return df, None

    def get_deltas_summary(self, instrument_id: str) -> DeltasSummaryResponse:
        files = self.list_files("order_book_deltas", instrument_id)
        signature = self._signature_for(files)
        key = build_cache_key("deltas_summary", instrument_id=instrument_id)
        cached = self.cache.get_model(key, signature, DeltasSummaryResponse)
        if cached is not None:
            return cached

        if not files:
            response = DeltasSummaryResponse(
                instrument_id=instrument_id,
                instrument_type=instrument_type_from_id(instrument_id),
                generated_at=utc_now_iso(),
                present=False,
            )
            self.cache.set_model(key, signature, response)
            return response

        try:
            dataset = ds.dataset([str(p) for p in files], format="parquet")
            ts_field = self._resolve_ts_field(dataset.schema.names)
            row_count = dataset.count_rows()
            ts_table = dataset.to_table(columns=[ts_field])
        except Exception as exc:
            response = DeltasSummaryResponse(
                instrument_id=instrument_id,
                instrument_type=instrument_type_from_id(instrument_id),
                generated_at=utc_now_iso(),
                present=True,
                file_count=len(files),
                error=str(exc),
            )
            self.cache.set_model(key, signature, response)
            return response

        if ts_table.num_rows == 0:
            response = DeltasSummaryResponse(
                instrument_id=instrument_id,
                instrument_type=instrument_type_from_id(instrument_id),
                generated_at=utc_now_iso(),
                present=True,
                file_count=len(files),
                total_rows=0,
            )
            self.cache.set_model(key, signature, response)
            return response

        table_dict = ts_table.to_pydict()
        order = _sort_table_by_ts(table_dict, ts_field)
        ts_values = np.asarray(table_dict[ts_field], dtype=np.int64)[order]
        gaps = compute_gap_entries(ts_values.tolist(), top_n=10)
        duration_seconds = float(ts_values[-1] - ts_values[0]) / 1_000_000_000 if len(ts_values) > 1 else 0.0
        threshold_ns = 300 * 1_000_000_000
        session_break_count = int(np.sum(np.diff(ts_values) >= threshold_ns)) if len(ts_values) > 1 else 0

        response = DeltasSummaryResponse(
            instrument_id=instrument_id,
            instrument_type=instrument_type_from_id(instrument_id),
            generated_at=utc_now_iso(),
            present=True,
            file_count=len(files),
            total_rows=int(row_count),
            ts_event_min_ns=int(ts_values[0]),
            ts_event_min_iso=ns_to_iso(int(ts_values[0])),
            ts_event_max_ns=int(ts_values[-1]),
            ts_event_max_iso=ns_to_iso(int(ts_values[-1])),
            duration_seconds=duration_seconds,
            max_gap_seconds=gaps[0].gap_seconds if gaps else None,
            session_break_count=session_break_count,
            top_gaps=gaps,
        )
        self.cache.set_model(key, signature, response)
        return response

    def get_deltas(
        self,
        instrument_id: str,
        *,
        from_value: str | int | None = None,
        to_value: str | int | None = None,
        mode: Literal["raw", "agg"] = "raw",
        bucket_s: int = 60,
        max_points: int = 10_000,
    ) -> DeltasResponse:
        from_ns = parse_time_value(from_value)
        to_ns = parse_time_value(to_value)
        files = self.list_files("order_book_deltas", instrument_id)
        signature = self._signature_for(files)
        key = build_cache_key(
            "deltas",
            instrument_id=instrument_id,
            from_ns=from_ns,
            to_ns=to_ns,
            mode=mode,
            bucket_s=bucket_s,
            max_points=max_points,
        )
        cached = self.cache.get_model(key, signature, DeltasResponse)
        if cached is not None:
            return cached

        df, error = self._delta_dataframe(instrument_id, from_ns, to_ns)
        if error is not None:
            return DeltasResponse(
                instrument_id=instrument_id,
                instrument_type=instrument_type_from_id(instrument_id),
                mode=mode,
                bucket_s=bucket_s,
                max_points=max_points,
                generated_at=utc_now_iso(),
                error=error,
            )

        if df.empty:
            response = DeltasResponse(
                instrument_id=instrument_id,
                instrument_type=instrument_type_from_id(instrument_id),
                mode=mode,
                bucket_s=bucket_s,
                max_points=max_points,
                total_rows=0,
                returned_points=0,
                from_ns=from_ns,
                from_iso=ns_to_iso(from_ns),
                to_ns=to_ns,
                to_iso=ns_to_iso(to_ns),
                generated_at=utc_now_iso(),
            )
            self.cache.set_model(key, signature, response)
            return response

        points: list[DeltaSeriesPoint] = []
        if mode == "agg":
            bucket_ns = max(1, bucket_s) * 1_000_000_000
            working = df.copy()
            working["bucket_ns"] = (working["ts_event_ns"] // bucket_ns) * bucket_ns
            working["is_add"] = working["action"] == "ADD"
            working["is_update"] = working["action"] == "UPDATE"
            working["is_delete"] = working["action"] == "DELETE"
            working["is_clear"] = working["action"] == "CLEAR"
            grouped = working.groupby("bucket_ns", as_index=False).agg(
                delta_count=("ts_event_ns", "count"),
                add_count=("is_add", "sum"),
                update_count=("is_update", "sum"),
                delete_count=("is_delete", "sum"),
                clear_count=("is_clear", "sum"),
            )
            order = _evenly_spaced_indices(len(grouped), max_points)
            for row in grouped.iloc[order].itertuples(index=False):
                ts_event_ns = int(row.bucket_ns)
                points.append(
                    DeltaSeriesPoint(
                        ts_event_ns=ts_event_ns,
                        ts_event_iso=ns_to_iso(ts_event_ns) or "",
                        delta_count=int(row.delta_count),
                        add_count=int(row.add_count),
                        update_count=int(row.update_count),
                        delete_count=int(row.delete_count),
                        clear_count=int(row.clear_count),
                    ),
                )
        else:
            order = _evenly_spaced_indices(len(df), max_points)
            for row in df.iloc[order].itertuples(index=False):
                points.append(
                    DeltaSeriesPoint(
                        ts_event_ns=int(row.ts_event_ns),
                        ts_event_iso=ns_to_iso(int(row.ts_event_ns)) or "",
                        action=str(row.action),
                        side=str(row.side),
                        price=float(row.price) if not np.isnan(row.price) else None,
                        size=float(row.size) if not np.isnan(row.size) else None,
                        flags=int(row.flags),
                        sequence=int(row.sequence),
                    ),
                )

        response = DeltasResponse(
            instrument_id=instrument_id,
            instrument_type=instrument_type_from_id(instrument_id),
            mode=mode,
            bucket_s=bucket_s,
            max_points=max_points,
            total_rows=len(df),
            returned_points=len(points),
            from_ns=int(df["ts_event_ns"].iloc[0]),
            from_iso=ns_to_iso(int(df["ts_event_ns"].iloc[0])),
            to_ns=int(df["ts_event_ns"].iloc[-1]),
            to_iso=ns_to_iso(int(df["ts_event_ns"].iloc[-1])),
            generated_at=utc_now_iso(),
            points=points,
        )
        self.cache.set_model(key, signature, response)
        return response

    # ── Readiness ───────────────────────────────────────────────────────────

    def get_readiness(self, instrument_id: str) -> ReadinessResponse:
        trade_files = self.list_files("trade_tick", instrument_id)
        delta_files = self.list_files("order_book_deltas", instrument_id)
        depth_files = self.list_files("order_book_depths", instrument_id)
        signature = self._signature_for(trade_files, delta_files, depth_files)
        key = build_cache_key("readiness", instrument_id=instrument_id)
        cached = self.cache.get_model(key, signature, ReadinessResponse)
        if cached is not None:
            return cached

        instrument_type = instrument_type_from_id(instrument_id)
        has_trade = len(trade_files) > 0
        has_deltas = len(delta_files) > 0
        has_depths = len(depth_files) > 0
        delta_first_only = has_deltas and not has_depths

        # Gather coverage stats for each present data type
        trade_row_count = 0
        trade_duration: float | None = None
        trade_max_gap: float | None = None
        if has_trade:
            cov = self._coverage_for_type(data_type="trade_tick", instrument_id=instrument_id, from_ns=None, to_ns=None)
            trade_row_count = cov.row_count
            trade_duration = cov.duration_seconds
            trade_max_gap = cov.max_gap_seconds

        delta_row_count = 0
        delta_duration: float | None = None
        delta_max_gap: float | None = None
        session_break_count = 0
        if has_deltas:
            cov = self._coverage_for_type(data_type="order_book_deltas", instrument_id=instrument_id, from_ns=None, to_ns=None)
            delta_row_count = cov.row_count
            delta_duration = cov.duration_seconds
            delta_max_gap = cov.max_gap_seconds
            session_break_count = cov.session_break_count

        depth_row_count = 0
        if has_depths:
            cov = self._coverage_for_type(data_type="order_book_depths", instrument_id=instrument_id, from_ns=None, to_ns=None)
            depth_row_count = cov.row_count

        # Compute backtest readiness as data-type completeness only.
        limitations: list[str] = []
        if not has_trade:
            limitations.append("No trade_tick data")
        if not has_deltas:
            limitations.append("No order_book_deltas data")
        if not has_depths:
            limitations.append("No optional order_book_depths (depth10)")

        is_consumable = has_trade or has_deltas
        is_backtest_ready = has_trade and has_deltas and delta_row_count > 0

        # Note: fenced_range_count / desync_count / resync_count are not available
        # without loading per-instrument report files. Pass 0 so the live-API score
        # is consistent with the audit formula (which does penalise when those exist).
        score, breakdown = compute_readiness_breakdown(
            has_trade_tick=has_trade,
            has_order_book_deltas=has_deltas,
            has_order_book_depths=has_depths,
            trade_row_count=trade_row_count,
            delta_row_count=delta_row_count,
            trade_max_gap_seconds=trade_max_gap,
            delta_max_gap_seconds=delta_max_gap,
            fenced_range_count=0,
            desync_count=0,
            resync_count=0,
            session_break_count=session_break_count,
        )

        result = ReadinessResult(
            instrument_id=instrument_id,
            instrument_type=instrument_type,
            has_trade_tick=has_trade,
            has_order_book_deltas=has_deltas,
            has_order_book_depths=has_depths,
            delta_first_only=delta_first_only,
            is_consumable=is_consumable,
            is_backtest_ready=is_backtest_ready,
            trade_row_count=trade_row_count,
            delta_row_count=delta_row_count,
            depth_row_count=depth_row_count,
            trade_duration_seconds=trade_duration,
            delta_duration_seconds=delta_duration,
            trade_max_gap_seconds=trade_max_gap,
            delta_max_gap_seconds=delta_max_gap,
            session_break_count=session_break_count,
            backtest_readiness_score=score,
            readiness_status=readiness_status_for_score(score),
            readiness_score=score,
            score_breakdown=breakdown,
            limitations=limitations,
        )

        response = ReadinessResponse(
            instrument_id=instrument_id,
            instrument_type=instrument_type,
            generated_at=utc_now_iso(),
            readiness=result,
        )
        self.cache.set_model(key, signature, response)
        return response

    # ── L2 depth10 (secondary / optional) ───────────────────────────────────

    def _full_l2_index(self, instrument_id: str) -> np.ndarray:
        files = self.list_files("order_book_depths", instrument_id)
        signature = self._signature_for(files)
        cached = self._l2_index_cache.get(instrument_id)
        if cached is not None and cached[0] == signature:
            return cached[1]

        dataset = self._dataset("order_book_depths", instrument_id)
        if dataset is None:
            values = np.asarray([], dtype=np.int64)
            self._l2_index_cache[instrument_id] = (signature, values)
            return values

        ts_field = self._resolve_ts_field(dataset.schema.names)
        table = dataset.to_table(columns=[ts_field])
        values = np.sort(np.asarray(table.column(ts_field).to_numpy(), dtype=np.int64))
        self._l2_index_cache[instrument_id] = (signature, values)
        return values

    def _l2_snapshots(
        self,
        instrument_id: str,
        *,
        from_ns: int | None,
        to_ns: int | None,
        with_global_index: bool = False,
    ) -> tuple[list[Any], str | None]:
        dataset = self._dataset("order_book_depths", instrument_id)
        if dataset is None:
            return [], None

        try:
            ts_field = self._resolve_ts_field(dataset.schema.names)
            columns = [ts_field] + self._resolve_depth_columns(dataset.schema.names)
            table = dataset.to_table(columns=columns, filter=_time_filter(ts_field, from_ns, to_ns))
        except Exception as exc:
            return [], str(exc)

        if table.num_rows == 0:
            return [], None

        table_dict = table.to_pydict()
        order = _sort_table_by_ts(table_dict, ts_field)
        ts_values = np.asarray(table_dict[ts_field], dtype=np.int64)[order]
        global_index_map: dict[int, int] = {}
        if with_global_index:
            full_index = self._full_l2_index(instrument_id)
            global_index_map = {int(value): idx for idx, value in enumerate(full_index.tolist())}

        snapshots = []
        for position in order.tolist():
            record = {column: table_dict[column][position] for column in table_dict}
            ts_event_ns = int(record[ts_field])
            index = global_index_map.get(ts_event_ns, len(snapshots)) if with_global_index else len(snapshots)
            snapshots.append(parsed_snapshot_from_record(record, index=index))
        return snapshots, None

    def get_l2_timeseries(
        self,
        instrument_id: str,
        *,
        from_value: str | int | None = None,
        to_value: str | int | None = None,
        mode: Literal["raw", "agg"] = "raw",
        bucket_s: int = 60,
        max_points: int = 10_000,
    ) -> L2TimeseriesResponse:
        from_ns = parse_time_value(from_value)
        to_ns = parse_time_value(to_value)
        files = self.list_files("order_book_depths", instrument_id)
        signature = self._signature_for(files)
        key = build_cache_key(
            "l2_timeseries",
            instrument_id=instrument_id,
            from_ns=from_ns,
            to_ns=to_ns,
            mode=mode,
            bucket_s=bucket_s,
            max_points=max_points,
        )
        cached = self.cache.get_model(key, signature, L2TimeseriesResponse)
        if cached is not None:
            return cached

        snapshots, error = self._l2_snapshots(instrument_id, from_ns=from_ns, to_ns=to_ns)
        if error is not None:
            return L2TimeseriesResponse(
                instrument_id=instrument_id,
                instrument_type=instrument_type_from_id(instrument_id),
                mode=mode,
                bucket_s=bucket_s,
                max_points=max_points,
                generated_at=utc_now_iso(),
                error=error,
            )

        if not snapshots:
            response = L2TimeseriesResponse(
                instrument_id=instrument_id,
                instrument_type=instrument_type_from_id(instrument_id),
                mode=mode,
                bucket_s=bucket_s,
                max_points=max_points,
                total_rows=0,
                returned_points=0,
                from_ns=from_ns,
                from_iso=ns_to_iso(from_ns),
                to_ns=to_ns,
                to_iso=ns_to_iso(to_ns),
                generated_at=utc_now_iso(),
            )
            self.cache.set_model(key, signature, response)
            return response

        frame = pd.DataFrame(
            {
                "ts_event_ns": [snapshot.ts_event_ns for snapshot in snapshots],
                "best_bid": [snapshot.best_bid for snapshot in snapshots],
                "best_ask": [snapshot.best_ask for snapshot in snapshots],
                "spread": [snapshot.spread for snapshot in snapshots],
                "mid": [snapshot.mid for snapshot in snapshots],
                "is_crossed": [snapshot.is_crossed for snapshot in snapshots],
                "is_sorted_ok": [snapshot.is_sorted_ok for snapshot in snapshots],
                "has_negative_qty": [snapshot.has_negative_qty for snapshot in snapshots],
                "has_zero_qty": [snapshot.has_zero_qty for snapshot in snapshots],
                "has_empty_side": [snapshot.has_empty_side for snapshot in snapshots],
                "bad_count": [1 if snapshot.issues else 0 for snapshot in snapshots],
            },
        )

        points: list[L2TimeseriesPoint] = []
        if mode == "agg":
            bucket_ns = max(1, bucket_s) * 1_000_000_000
            working = frame.copy()
            working["bucket_ns"] = (working["ts_event_ns"] // bucket_ns) * bucket_ns
            grouped = working.groupby("bucket_ns", as_index=False).agg(
                best_bid=("best_bid", "last"),
                best_ask=("best_ask", "last"),
                spread=("spread", "mean"),
                mid=("mid", "mean"),
                update_count=("ts_event_ns", "count"),
                crossed_count=("is_crossed", "sum"),
                empty_count=("has_empty_side", "sum"),
                bad_count=("bad_count", "sum"),
            )
            order = _evenly_spaced_indices(len(grouped), max_points)
            for row in grouped.iloc[order].itertuples(index=False):
                ts_event_ns = int(row.bucket_ns)
                points.append(
                    L2TimeseriesPoint(
                        ts_event_ns=ts_event_ns,
                        ts_event_iso=ns_to_iso(ts_event_ns) or "",
                        best_bid=float(row.best_bid) if not pd.isna(row.best_bid) else None,
                        best_ask=float(row.best_ask) if not pd.isna(row.best_ask) else None,
                        spread=float(row.spread) if not pd.isna(row.spread) else None,
                        mid=float(row.mid) if not pd.isna(row.mid) else None,
                        update_count=int(row.update_count),
                        update_rate_per_minute=(float(row.update_count) * 60.0 / bucket_s),
                        crossed_count=int(row.crossed_count),
                        empty_count=int(row.empty_count),
                        bad_count=int(row.bad_count),
                    ),
                )
        else:
            order = _evenly_spaced_indices(len(frame), max_points)
            for row in frame.iloc[order].itertuples(index=False):
                points.append(
                    L2TimeseriesPoint(
                        ts_event_ns=int(row.ts_event_ns),
                        ts_event_iso=ns_to_iso(int(row.ts_event_ns)) or "",
                        best_bid=float(row.best_bid) if row.best_bid is not None else None,
                        best_ask=float(row.best_ask) if row.best_ask is not None else None,
                        spread=float(row.spread) if row.spread is not None else None,
                        mid=float(row.mid) if row.mid is not None else None,
                        is_crossed=bool(row.is_crossed),
                        is_sorted_ok=bool(row.is_sorted_ok),
                        has_negative_qty=bool(row.has_negative_qty),
                        has_zero_qty=bool(row.has_zero_qty),
                        has_empty_side=bool(row.has_empty_side),
                    ),
                )

        response = L2TimeseriesResponse(
            instrument_id=instrument_id,
            instrument_type=instrument_type_from_id(instrument_id),
            mode=mode,
            bucket_s=bucket_s,
            max_points=max_points,
            total_rows=len(frame),
            returned_points=len(points),
            from_ns=int(frame["ts_event_ns"].iloc[0]),
            from_iso=ns_to_iso(int(frame["ts_event_ns"].iloc[0])),
            to_ns=int(frame["ts_event_ns"].iloc[-1]),
            to_iso=ns_to_iso(int(frame["ts_event_ns"].iloc[-1])),
            generated_at=utc_now_iso(),
            points=points,
        )
        self.cache.set_model(key, signature, response)
        return response

    def get_l2_snapshot(
        self,
        instrument_id: str,
        *,
        ts_value: str | int | None = None,
        index: int | None = None,
        context_before: int = 5,
        context_after: int = 5,
    ) -> L2SnapshotResponse:
        requested_ts_ns = parse_time_value(ts_value)
        files = self.list_files("order_book_depths", instrument_id)
        signature = self._signature_for(files)
        key = build_cache_key(
            "l2_snapshot",
            instrument_id=instrument_id,
            ts_ns=requested_ts_ns,
            index=index,
            context_before=context_before,
            context_after=context_after,
        )
        cached = self.cache.get_model(key, signature, L2SnapshotResponse)
        if cached is not None:
            return cached

        full_index = self._full_l2_index(instrument_id)
        if full_index.size == 0:
            response = L2SnapshotResponse(
                instrument_id=instrument_id,
                instrument_type=instrument_type_from_id(instrument_id),
                total_snapshots=0,
                requested_index=index,
                requested_ts_ns=requested_ts_ns,
                requested_ts_iso=ns_to_iso(requested_ts_ns),
                generated_at=utc_now_iso(),
                error="Nincs order_book_depths adat ehhez az instrumenthez.",
            )
            self.cache.set_model(key, signature, response)
            return response

        if index is not None:
            resolved_index = max(0, min(int(index), len(full_index) - 1))
        elif requested_ts_ns is not None:
            insertion = int(np.searchsorted(full_index, requested_ts_ns, side="left"))
            candidates = [max(0, min(insertion, len(full_index) - 1))]
            if insertion > 0:
                candidates.append(insertion - 1)
            resolved_index = min(candidates, key=lambda candidate: abs(int(full_index[candidate]) - requested_ts_ns))
        else:
            resolved_index = 0

        start_index = max(0, resolved_index - max(0, context_before))
        end_index = min(len(full_index) - 1, resolved_index + max(0, context_after))
        from_ns = int(full_index[start_index])
        to_ns = int(full_index[end_index])
        snapshots, error = self._l2_snapshots(
            instrument_id,
            from_ns=from_ns,
            to_ns=to_ns,
            with_global_index=True,
        )
        if error is not None:
            return L2SnapshotResponse(
                instrument_id=instrument_id,
                instrument_type=instrument_type_from_id(instrument_id),
                total_snapshots=len(full_index),
                requested_index=index,
                requested_ts_ns=requested_ts_ns,
                requested_ts_iso=ns_to_iso(requested_ts_ns),
                generated_at=utc_now_iso(),
                error=error,
            )

        snapshot_map = {snapshot.index: snapshot for snapshot in snapshots}
        resolved_snapshot = snapshot_map.get(resolved_index)
        context = [parsed_to_summary(snapshot) for snapshot in snapshots]
        response = L2SnapshotResponse(
            instrument_id=instrument_id,
            instrument_type=instrument_type_from_id(instrument_id),
            total_snapshots=len(full_index),
            resolved_index=resolved_index,
            requested_index=index,
            requested_ts_ns=requested_ts_ns,
            requested_ts_iso=ns_to_iso(requested_ts_ns),
            generated_at=utc_now_iso(),
            snapshot=parsed_to_snapshot(resolved_snapshot) if resolved_snapshot is not None else None,
            context=context,
            error=None if resolved_snapshot is not None else "A k\u00e9rt snapshot nem tal\u00e1lhat\u00f3.",
        )
        self.cache.set_model(key, signature, response)
        return response

    def get_l2_quality(
        self,
        instrument_id: str,
        *,
        from_value: str | int | None = None,
        to_value: str | int | None = None,
    ) -> L2QualityResponse:
        from_ns = parse_time_value(from_value)
        to_ns = parse_time_value(to_value)
        files = self.list_files("order_book_depths", instrument_id)
        signature = self._signature_for(files)
        key = build_cache_key("l2_quality", instrument_id=instrument_id, from_ns=from_ns, to_ns=to_ns)
        cached = self.cache.get_model(key, signature, L2QualityResponse)
        if cached is not None:
            return cached

        snapshots, error = self._l2_snapshots(
            instrument_id,
            from_ns=from_ns,
            to_ns=to_ns,
            with_global_index=True,
        )
        if error is not None:
            return L2QualityResponse(
                instrument_id=instrument_id,
                instrument_type=instrument_type_from_id(instrument_id),
                from_ns=from_ns,
                from_iso=ns_to_iso(from_ns),
                to_ns=to_ns,
                to_iso=ns_to_iso(to_ns),
                generated_at=utc_now_iso(),
                error=error,
            )

        response = quality_from_snapshots(
            instrument_id=instrument_id,
            instrument_type=instrument_type_from_id(instrument_id),
            snapshots=snapshots,
            from_ns=from_ns,
            to_ns=to_ns,
        )
        self.cache.set_model(key, signature, response)
        return response

    # ── Export ───────────────────────────────────────────────────────────────

    def export_trades_csv(self, instrument_id: str, *, from_value: str | int | None = None, to_value: str | int | None = None) -> bytes:
        from_ns = parse_time_value(from_value)
        to_ns = parse_time_value(to_value)
        df, error = self._trade_dataframe(instrument_id, from_ns, to_ns)
        if error is not None:
            raise ValueError(error)

        buffer = io.StringIO()
        writer = csv.writer(buffer)
        writer.writerow(["ts_event_ns", "ts_event_iso", "price", "size", "aggressor_side", "trade_id"])
        for row in df.itertuples(index=False):
            writer.writerow(
                [
                    int(row.ts_event_ns),
                    ns_to_iso(int(row.ts_event_ns)),
                    float(row.price),
                    float(row.size),
                    row.aggressor_side,
                    row.trade_id,
                ],
            )
        return buffer.getvalue().encode("utf-8")

    def export_l2_csv(self, instrument_id: str, *, from_value: str | int | None = None, to_value: str | int | None = None) -> bytes:
        from_ns = parse_time_value(from_value)
        to_ns = parse_time_value(to_value)
        snapshots, error = self._l2_snapshots(instrument_id, from_ns=from_ns, to_ns=to_ns, with_global_index=True)
        if error is not None:
            raise ValueError(error)

        buffer = io.StringIO()
        writer = csv.writer(buffer)
        writer.writerow(
            [
                "index",
                "ts_event_ns",
                "ts_event_iso",
                "best_bid",
                "best_ask",
                "spread",
                "mid",
                "is_crossed",
                "is_sorted_ok",
                "has_negative_qty",
                "has_zero_qty",
                "has_empty_side",
                "bids",
                "asks",
                "bid_sizes",
                "ask_sizes",
                "issues",
            ],
        )
        for snapshot in snapshots:
            writer.writerow(
                [
                    snapshot.index,
                    snapshot.ts_event_ns,
                    ns_to_iso(snapshot.ts_event_ns),
                    snapshot.best_bid,
                    snapshot.best_ask,
                    snapshot.spread,
                    snapshot.mid,
                    snapshot.is_crossed,
                    snapshot.is_sorted_ok,
                    snapshot.has_negative_qty,
                    snapshot.has_zero_qty,
                    snapshot.has_empty_side,
                    json.dumps(snapshot.bids),
                    json.dumps(snapshot.asks),
                    json.dumps(snapshot.bid_sizes),
                    json.dumps(snapshot.ask_sizes),
                    json.dumps(snapshot.issues),
                ],
            )
        return buffer.getvalue().encode("utf-8")

    def export_bundle_json(self, instrument_id: str, *, from_value: str | int | None = None, to_value: str | int | None = None) -> bytes:
        coverage = self.get_coverage(instrument_id, from_value=from_value, to_value=to_value)
        trades = self.get_trades(
            instrument_id,
            from_value=from_value,
            to_value=to_value,
            mode="raw",
            max_points=20_000,
        )
        deltas_summary = self.get_deltas_summary(instrument_id)
        readiness = self.get_readiness(instrument_id)
        l2_quality = self.get_l2_quality(instrument_id, from_value=from_value, to_value=to_value)
        snapshots, error = self._l2_snapshots(
            instrument_id,
            from_ns=parse_time_value(from_value),
            to_ns=parse_time_value(to_value),
            with_global_index=True,
        )
        if error is not None:
            raise ValueError(error)

        payload = {
            "instrument_id": instrument_id,
            "instrument_type": instrument_type_from_id(instrument_id),
            "generated_at": utc_now_iso(),
            "coverage": coverage.model_dump(mode="json"),
            "readiness": readiness.model_dump(mode="json"),
            "deltas_summary": deltas_summary.model_dump(mode="json"),
            "trades": trades.model_dump(mode="json"),
            "l2_quality": l2_quality.model_dump(mode="json"),
            "l2_snapshots": [parsed_to_snapshot(snapshot).model_dump(mode="json") for snapshot in snapshots],
        }
        return json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
