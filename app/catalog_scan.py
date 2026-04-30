from __future__ import annotations

import heapq
import json
import os
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pyarrow.parquet as pq

from .l2_checks import compute_l2_quality_score, estimate_missing_ratio, run_l2_checks
from .models import (
    AuditInstrumentResult,
    AuditResponse,
    AuditSummary,
    CatalogWarning,
    ChartPoint,
    DataTypeAuditStats,
    FencedRange,
    GapEntry,
    InstrumentCoverage,
    InstrumentInventoryItem,
    InstrumentTypeSummary,
    InventoryResponse,
    L2CheckResult,
    QualityOffenderItem,
    ReadinessOffenderItem,
    ReadinessResult,
    ReportContext,
    ResyncEvent,
    SessionBoundary,
)
from .query import CatalogQueryService

try:
    from nautilus_trader.persistence.catalog import ParquetDataCatalog
except Exception:  # pragma: no cover
    ParquetDataCatalog = None  # type: ignore[assignment]


INSTRUMENT_TYPE_DIRS: tuple[str, ...] = ("crypto_perpetual", "currency_pair")
EVENT_DATA_TYPES: tuple[str, ...] = ("trade_tick", "order_book_deltas", "order_book_depths")
TIMESTAMP_CANDIDATES: tuple[str, ...] = (
    "ts_event",
    "ts_init",
    "event_ts",
    "ts",
    "timestamp",
    "time",
    "ts_recv",
)

ProgressCallback = Callable[[str, int, int, str], None]


def _utc_now_iso() -> str:
    return datetime.now(tz=timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _ns_to_iso(value_ns: int | None) -> str | None:
    if value_ns is None:
        return None
    seconds, nanos = divmod(value_ns, 1_000_000_000)
    dt = datetime.fromtimestamp(seconds, tz=timezone.utc)
    return f"{dt.strftime('%Y-%m-%dT%H:%M:%S')}.{nanos:09d}Z"


def _snake_case(name: str) -> str:
    return re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()


def _relative_file_label(catalog_root: Path, file_path: Path) -> str:
    try:
        return str(file_path.relative_to(catalog_root))
    except ValueError:
        return str(file_path)


def _dedupe_warnings(warnings: list[CatalogWarning]) -> list[CatalogWarning]:
    seen: set[tuple[str, str, str | None]] = set()
    unique: list[CatalogWarning] = []
    for warning in warnings:
        key = (warning.code, warning.message, warning.path)
        if key not in seen:
            seen.add(key)
            unique.append(warning)
    return unique


from .scoring import compute_readiness_breakdown, compute_readiness_score, readiness_status_for_score, readiness_status_for_presence


def _converter_key_to_instrument_id(key: str) -> str:
    """Map 'BINANCE_SPOT/AAVEUSDT' -> 'AAVEUSDT.BINANCE' and 'BINANCE_USDTF/AAVEUSDT' -> 'AAVEUSDT-PERP.BINANCE'."""
    parts = key.split("/", 1)
    if len(parts) != 2:
        return key
    venue, symbol = parts
    exchange = venue.split("_")[0]  # e.g. BINANCE from BINANCE_SPOT
    is_perp = any(tag in venue for tag in ("USDTF", "PERP", "FUTURES", "SWAP"))
    return f"{symbol}-PERP.{exchange}" if is_perp else f"{symbol}.{exchange}"


def _coerce_int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def _coerce_str_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [str(item) for item in value]
    if isinstance(value, dict):
        return [str(key) for key in value]
    return [str(value)]


def _normalize_report_symbol(value: str) -> str:
    if "/" in value:
        return _converter_key_to_instrument_id(value)
    return value


def _selected_converter_report(
    converter_reports_dir: Path | None,
    *,
    report_date: str | None = None,
) -> tuple[dict, Path | None, str | None, list[str]]:
    """Load the selected converter report.

    Latest YYYY-MM-DD.json wins when no report date is requested.
    """
    warnings: list[str] = []
    if converter_reports_dir is None:
        return {}, None, None, warnings
    if not converter_reports_dir.exists():
        warnings.append(f"Converter report directory does not exist: {converter_reports_dir}")
        return {}, None, None, warnings
    if report_date:
        candidate = converter_reports_dir / f"{report_date}.json"
        if not candidate.exists():
            warnings.append(f"Converter report for {report_date} was not found in {converter_reports_dir}")
            return {}, None, report_date, warnings
        selected = candidate
    else:
        json_files = sorted(path for path in converter_reports_dir.glob("*.json") if re.match(r"\d{4}-\d{2}-\d{2}\.json$", path.name))
        if not json_files:
            json_files = sorted(converter_reports_dir.glob("*.json"))
        if not json_files:
            warnings.append(f"No converter reports found in {converter_reports_dir}")
            return {}, None, None, warnings
        selected = json_files[-1]
        report_date = selected.stem
    try:
        return json.loads(selected.read_text(encoding="utf-8")), selected, report_date, warnings
    except Exception as exc:
        warnings.append(f"Converter report could not be read: {exc}")
        return {}, selected, report_date, warnings


def _load_converter_report(converter_reports_dir: Path, report_date: str | None = None) -> dict:
    """Backward-compatible helper returning only the parsed report dict."""
    raw, _, _, _ = _selected_converter_report(converter_reports_dir, report_date=report_date)
    return raw


def _converter_report_timestamp(raw: dict) -> str | None:
    for key in ("timestamp", "generated_at", "created_at", "finished_at", "ts"):
        value = raw.get(key)
        if value:
            return str(value)
    return None


def _converter_report_warnings(raw: dict, discovery_warnings: list[str]) -> list[str]:
    warnings = list(discovery_warnings)
    for key in ("warnings", "converter_warnings", "errors"):
        warnings.extend(_coerce_str_list(raw.get(key)))
    return warnings


def _converter_report_paths(raw: dict, selected_path: Path | None) -> list[str]:
    paths = _coerce_str_list(raw.get("report_paths"))
    extra = raw.get("convert_report_extra_path")
    if extra:
        paths.append(str(extra))
    if selected_path is not None:
        paths.insert(0, str(selected_path))
    seen: set[str] = set()
    result: list[str] = []
    for path in paths:
        if path and path not in seen:
            seen.add(path)
            result.append(path)
    return result


def _converter_catalog_matches(raw: dict, catalog_root: Path) -> bool | None:
    report_catalog_root = raw.get("catalog_root")
    if not report_catalog_root:
        return None
    try:
        return Path(str(report_catalog_root)).expanduser().resolve() == catalog_root
    except Exception:
        return str(report_catalog_root) == str(catalog_root)


def _converter_bad_lines(raw: dict) -> int:
    for key in ("bad_lines", "bad_line_count", "malformed_lines"):
        if key in raw:
            return _coerce_int(raw.get(key))
    return 0


def _converter_global_count(raw: dict, *keys: str) -> int:
    for key in keys:
        if key in raw:
            value = raw.get(key)
            if isinstance(value, list):
                return len(value)
            return _coerce_int(value)
    return 0


def _converter_missing_symbol_set(raw: dict) -> set[str]:
    keys = (
        "missing_raw_symbols",
        "missing_converted_symbols",
        "missing_symbols",
        "unconverted_symbols",
    )
    result: set[str] = set()
    for key in keys:
        result.update(_normalize_report_symbol(item) for item in _coerce_str_list(raw.get(key)))
    return result


def _converter_partial_unreadable_symbol_set(raw: dict) -> set[str]:
    keys = (
        "partial_symbols",
        "unreadable_symbols",
        "partial_unreadable_symbols",
        "partial_data_symbols",
    )
    result: set[str] = set()
    for key in keys:
        result.update(_normalize_report_symbol(item) for item in _coerce_str_list(raw.get(key)))
    return result


def _converter_row_count_for_instrument(raw: dict, instrument_id: str, data_type: str) -> int | None:
    """Best-effort per-symbol row count from CryptoRecorder convert reports."""
    keys_by_type = {
        "trade_tick": ("per_symbol_trade", "per_symbol_trades", "trade_tick"),
        "order_book_deltas": ("per_symbol_deltas", "per_symbol_delta", "per_symbol_depth", "order_book_deltas"),
        "order_book_depths": ("per_symbol_depth", "per_symbol_depth10", "order_book_depths"),
    }
    count_keys_by_type = {
        "trade_tick": ("ticks_written", "trades_written", "trade_ticks_written", "rows_written", "row_count", "count"),
        "order_book_deltas": ("deltas_written", "rows_written", "row_count", "count"),
        "order_book_depths": ("depth10_written", "depths_written", "rows_written", "row_count", "count"),
    }
    report_keys = keys_by_type.get(data_type, ())
    count_keys = count_keys_by_type.get(data_type, ("rows_written", "row_count", "count"))
    for report_key in report_keys:
        section = raw.get(report_key)
        if not isinstance(section, dict):
            continue
        for key, value in section.items():
            if _normalize_report_symbol(str(key)) != instrument_id:
                continue
            if isinstance(value, dict):
                for count_key in count_keys:
                    if count_key in value:
                        return _coerce_int(value.get(count_key))
            return _coerce_int(value)

    integrity = raw.get("conversion_integrity")
    if isinstance(integrity, dict):
        by_symbol = integrity.get("by_symbol") or integrity.get("per_symbol")
        if isinstance(by_symbol, dict):
            value = by_symbol.get(instrument_id)
            if value is None:
                for key, candidate in by_symbol.items():
                    if _normalize_report_symbol(str(key)) == instrument_id:
                        value = candidate
                        break
            if isinstance(value, dict):
                for count_key in count_keys:
                    if count_key in value:
                        return _coerce_int(value.get(count_key))
    return None


def _converter_readiness_classification(raw: dict, instrument_id: str) -> str | None:
    """Extract the converter's readiness classification for an instrument.

    Supports three formats:
    A) grouped-list: {"full_ready": ["ZECUSDT.BINANCE", ...], "l2_ready": [...]})
    B) direct map:  {"ZECUSDT.BINANCE": "full_ready"}
    C) dict value:  {"ZECUSDT.BINANCE": {"status": "full_ready"}}
    """
    section = raw.get("readiness_classification")
    if not isinstance(section, dict):
        return None

    # Detect grouped-list format: values are lists of instrument IDs
    first_value = next(iter(section.values())) if section else None
    if isinstance(first_value, list):
        # Invert: {status: [id, ...]} -> {id: status}
        for status, ids in section.items():
            if isinstance(ids, list):
                for raw_id in ids:
                    normalized = _normalize_report_symbol(str(raw_id))
                    if normalized == instrument_id or str(raw_id) == instrument_id:
                        return str(status)
        return None

    # Direct map or dict-value format
    value = section.get(instrument_id)
    if value is None:
        for key, candidate in section.items():
            if _normalize_report_symbol(str(key)) == instrument_id:
                value = candidate
                break
    if isinstance(value, dict):
        for key in ("status", "classification", "readiness_status"):
            if value.get(key):
                return str(value[key])
        return None
    if value is not None:
        return str(value)
    return None


def _build_converter_trade_presence(converter_report: dict) -> dict:
    """Extract trade-presence summary from the converter report.

    Returns a dict with:
      - instruments_with_trades: int
      - no_trade_instrument_ids: list[str]   (Nautilus-style IDs)
    """
    dp = converter_report.get("data_presence", {})
    if not dp:
        return {}
    instrument_count = int(dp.get("instruments_with_trades", 0))
    no_data_raw: list[str] = dp.get("no_data_list", [])
    # no_data_list may contain Nautilus IDs already (e.g. UTKUSDT.BINANCE)
    no_trade_ids = [str(x) for x in no_data_raw]
    return {
        "instruments_with_trades": instrument_count,
        "no_trade_instrument_ids": no_trade_ids,
    }


def _build_converter_fenced_map(converter_report: dict) -> dict[str, dict]:
    """Build instrument_id -> {count, by_reason, examples} from the converter report.

    Returns an empty dict when the report data has no per_symbol_fenced_ranges.
    """
    raw = converter_report.get("per_symbol_fenced_ranges", {})
    result: dict[str, dict] = {}
    for key, value in raw.items():
        instrument_id = _converter_key_to_instrument_id(key)
        count = int(value.get("fenced_ranges", 0))
        by_reason: dict[str, int] = {}
        for example in value.get("examples", []):
            reason = str(example.get("reason", "unknown"))
            by_reason[reason] = by_reason.get(reason, 0) + 1
        result[instrument_id] = {
            "count": count,
            "by_reason": by_reason,
        }
    return result


def _load_report_context(report_dir: Path, instrument_id: str) -> ReportContext:
    report_path = report_dir / f"{instrument_id}.json"
    if not report_path.exists():
        return ReportContext(report_found=False)
    try:
        raw = json.loads(report_path.read_text(encoding="utf-8"))
    except Exception:
        return ReportContext(report_found=False)

    fenced_ranges = [
        FencedRange(
            start_ns=int(r.get("start_ns", 0)), start_iso=str(r.get("start_iso", "")),
            end_ns=int(r.get("end_ns", 0)), end_iso=str(r.get("end_iso", "")),
            reason=str(r.get("reason", "")),
        )
        for r in raw.get("fenced_ranges", [])
    ]
    session_boundaries = [
        SessionBoundary(
            ts_ns=int(b.get("ts_ns", 0)), ts_iso=str(b.get("ts_iso", "")),
            kind=b.get("kind", "start"), label=str(b.get("label", "")),
        )
        for b in raw.get("session_boundaries", [])
    ]
    resync_events = [
        ResyncEvent(
            ts_ns=int(e.get("ts_ns", 0)), ts_iso=str(e.get("ts_iso", "")),
            kind=e.get("kind", "resync"), detail=str(e.get("detail", "")),
        )
        for e in raw.get("resync_events", [])
    ]
    ss = sum(1 for e in resync_events if e.kind == "snapshot_seed")
    rc = sum(1 for e in resync_events if e.kind == "resync")
    dc = sum(1 for e in resync_events if e.kind == "desync")

    return ReportContext(
        report_found=True,
        snapshot_seed_count=int(raw.get("snapshot_seed_count", ss)),
        resync_count=int(raw.get("resync_count", rc)),
        desync_count=int(raw.get("desync_count", dc)),
        fenced_ranges=fenced_ranges,
        session_boundaries=session_boundaries,
        resync_events=resync_events,
        last_committed_update_id=raw.get("last_committed_update_id"),
        trade_id_diagnostics=raw.get("trade_id_diagnostics", []),
        converter_warnings=raw.get("converter_warnings", []),
    )


class CatalogScanner:
    def __init__(
        self,
        catalog_root: Path | str,
        cache_path: Path | str | None = None,
        converter_reports_dir: Path | str | None = None,
        convert_report_date: str | None = None,
    ) -> None:
        _raw = Path(catalog_root)
        if not _raw.expanduser().is_absolute():
            import warnings as _w
            _w.warn(
                f"catalog_root '{catalog_root}' is a relative path and will be resolved against "
                f"the current working directory ({Path.cwd()}). "
                f"Pass an absolute path (starting with '/') to avoid this.",
                UserWarning,
                stacklevel=2,
            )
        self.catalog_root = _raw.expanduser().resolve()
        project_root = Path(__file__).resolve().parent.parent
        default_cache = project_root / "state" / "web_audit_cache.json"
        self.cache_path = Path(cache_path).expanduser().resolve() if cache_path else default_cache
        self.convert_report_date = convert_report_date
        # Converter reports directory:
        # explicit arg > NAUTILUS_VIEWER_CONVERT_REPORT_DIR > sibling > app-local fallback.
        _env = os.getenv("NAUTILUS_VIEWER_CONVERT_REPORT_DIR") or os.getenv("NAUTILUS_CONVERTER_REPORTS_DIR")
        if converter_reports_dir is not None:
            self.converter_reports_dir: Path | None = Path(converter_reports_dir).expanduser().resolve()
        elif _env:
            self.converter_reports_dir = Path(_env).expanduser().resolve()
        else:
            sibling = self.catalog_root.parent / "convert_reports"
            app_local = project_root / "state" / "convert_reports"
            self.converter_reports_dir = sibling if sibling.exists() else app_local
        self._nautilus_catalog: Any | None = None
        self._nautilus_catalog_initialized = False
        self.query_service = CatalogQueryService(self.catalog_root)

    @property
    def data_root(self) -> Path:
        return self.catalog_root / "data"

    @property
    def report_dir(self) -> Path:
        return self.catalog_root / "reports"

    def _warning(self, code: str, message: str, path: Path | None = None) -> CatalogWarning:
        return CatalogWarning(code=code, message=message, path=str(path) if path else None)

    def _get_nautilus_catalog(self, warnings: list[CatalogWarning]) -> Any | None:
        if self._nautilus_catalog_initialized:
            return self._nautilus_catalog
        self._nautilus_catalog_initialized = True
        if ParquetDataCatalog is None:
            warnings.append(self._warning("nautilus_unavailable", "nautilus_trader is not installed; catalog listing falls back to filesystem."))
            return None
        try:
            self._nautilus_catalog = ParquetDataCatalog(str(self.catalog_root))
        except Exception as exc:
            warnings.append(self._warning("nautilus_catalog_init_failed", f"Could not open ParquetDataCatalog: {exc}", self.catalog_root))
            self._nautilus_catalog = None
        return self._nautilus_catalog

    def _load_instrument_index(self, warnings: list[CatalogWarning]) -> dict[str, str]:
        instrument_index: dict[str, str] = {}
        for instrument_type in INSTRUMENT_TYPE_DIRS:
            type_dir = self.data_root / instrument_type
            if not type_dir.exists():
                warnings.append(self._warning("missing_instrument_type_dir", f"Missing instrument type directory: {instrument_type}", type_dir))
                continue
            for instrument_dir in sorted(path for path in type_dir.iterdir() if path.is_dir()):
                instrument_index[instrument_dir.name] = instrument_type
        nautilus_catalog = self._get_nautilus_catalog(warnings)
        if nautilus_catalog is not None:
            try:
                for instrument in nautilus_catalog.instruments():
                    instrument_index[instrument.id.value] = _snake_case(type(instrument).__name__)
            except Exception as exc:
                warnings.append(self._warning("nautilus_instrument_list_failed", f"Nautilus instrument list could not be read: {exc}", self.catalog_root))
        return instrument_index

    def _event_file_map(self, data_type: str, warnings: list[CatalogWarning]) -> dict[str, list[Path]]:
        type_dir = self.data_root / data_type
        file_map: dict[str, list[Path]] = {}
        if not type_dir.exists():
            return file_map
        for instrument_dir in sorted(path for path in type_dir.iterdir() if path.is_dir()):
            parquet_files = sorted(path for path in instrument_dir.glob("*.parquet") if path.is_file())
            file_map[instrument_dir.name] = parquet_files
        return file_map

    def _infer_instrument_type(self, instrument_id: str) -> str:
        if "-PERP." in instrument_id:
            return "crypto_perpetual"
        return "currency_pair"

    def _collect_catalog_state(self) -> tuple[dict[str, str], dict[str, dict[str, list[Path]]], list[CatalogWarning]]:
        warnings: list[CatalogWarning] = []
        if not self.catalog_root.exists():
            _cwd = Path.cwd()
            _hint = (
                f" Hint: path appears to have been resolved relative to CWD ({_cwd}); "
                "did you forget the leading '/' in your --catalog argument?"
                if self.catalog_root.is_relative_to(_cwd)
                else ""
            )
            warnings.append(self._warning("missing_catalog_root", f"Catalog root does not exist.{_hint}", self.catalog_root))
        if not self.data_root.exists():
            warnings.append(self._warning("missing_data_root", "Catalog data directory does not exist.", self.data_root))
        instrument_index = self._load_instrument_index(warnings)
        event_maps = {data_type: self._event_file_map(data_type, warnings) for data_type in EVENT_DATA_TYPES}
        for file_map in event_maps.values():
            for instrument_id in file_map:
                instrument_index.setdefault(instrument_id, self._infer_instrument_type(instrument_id))
        return instrument_index, event_maps, _dedupe_warnings(warnings)

    def scan_inventory(self, search: str | None = None) -> InventoryResponse:
        instrument_index, event_maps, warnings = self._collect_catalog_state()
        search_term = search.lower().strip() if search else None
        instruments: list[InstrumentInventoryItem] = []
        grouped: dict[str, list[str]] = defaultdict(list)
        grouped_with_data: dict[str, int] = defaultdict(int)
        for instrument_id in sorted(instrument_index, key=lambda value: (instrument_index[value], value)):
            if search_term and search_term not in instrument_id.lower():
                continue
            instrument_type = instrument_index[instrument_id]
            grouped[instrument_type].append(instrument_id)
            coverage = {
                data_type: InstrumentCoverage(
                    data_type=data_type,
                    present=bool(event_maps[data_type].get(instrument_id)),
                    file_count=len(event_maps[data_type].get(instrument_id, [])),
                )
                for data_type in EVENT_DATA_TYPES
            }
            has_any_data = any(item.present for item in coverage.values())
            if has_any_data:
                grouped_with_data[instrument_type] += 1
            instruments.append(InstrumentInventoryItem(instrument_id=instrument_id, instrument_type=instrument_type, has_any_data=has_any_data, coverage=coverage))
        instrument_types = [
            InstrumentTypeSummary(instrument_type=it, instrument_count=len(names), with_any_data_count=grouped_with_data[it], instruments=names)
            for it, names in sorted(grouped.items())
        ]
        return InventoryResponse(catalog_root=str(self.catalog_root), generated_at=_utc_now_iso(), available_data_types=list(EVENT_DATA_TYPES), instrument_types=instrument_types, instruments=instruments, warnings=warnings)

    @staticmethod
    def _resolve_ts_for_parquet(schema_names: list[str]) -> str:
        """Return the best ts column name available in a parquet schema.

        Handles both regular schema names and row-group ``path_in_schema``
        values, which may include nested paths.
        """
        normalized = {name.lower(): name for name in schema_names}
        for candidate in TIMESTAMP_CANDIDATES:
            if candidate in normalized:
                return normalized[candidate]
        for name in schema_names:
            parts = re.split(r"[./]", name.lower())
            if any(part in TIMESTAMP_CANDIDATES for part in parts):
                return name
        for candidate in ("ts_event", "ts_init"):
            for name in schema_names:
                lname = name.lower()
                if lname.endswith(candidate) or candidate in lname:
                    return name
        for name in schema_names:
            lname = name.lower()
            if "timestamp" in lname or lname.endswith("ts"):
                return name
        raise KeyError(f"No ts_event-like column found in schema: {schema_names}")

    @staticmethod
    def _ts_array_to_int64(values: Any) -> np.ndarray:
        arr = np.asarray(values)
        if arr.size == 0:
            return np.asarray([], dtype=np.int64)
        if np.issubdtype(arr.dtype, np.datetime64):
            return arr.astype("datetime64[ns]").astype(np.int64)
        return arr.astype(np.int64)

    def _schema_names_for_parquet(self, parquet_file: pq.ParquetFile) -> list[str]:
        names: list[str] = []
        try:
            names.extend(parquet_file.schema_arrow.names)
        except Exception:
            pass
        metadata = parquet_file.metadata
        if metadata.num_row_groups > 0:
            rg0 = metadata.row_group(0)
            for ci in range(rg0.num_columns):
                path = rg0.column(ci).path_in_schema
                if path not in names:
                    names.append(path)
        return names

    def _metadata_ts_bounds_with_status(self, parquet_file: pq.ParquetFile, ts_col: str) -> tuple[int | None, int | None, bool]:
        metadata = parquet_file.metadata
        min_ts: int | None = None
        max_ts: int | None = None
        stats_available = False
        for rgi in range(metadata.num_row_groups):
            rg = metadata.row_group(rgi)
            for ci in range(rg.num_columns):
                col = rg.column(ci)
                if col.path_in_schema != ts_col and col.path_in_schema.split(".")[-1] != ts_col:
                    continue
                if col.statistics is None:
                    continue
                stats = col.statistics
                if not stats.has_min_max:
                    continue
                stats_available = True
                cmin, cmax = int(stats.min), int(stats.max)
                min_ts = cmin if min_ts is None else min(min_ts, cmin)
                max_ts = cmax if max_ts is None else max(max_ts, cmax)
        return min_ts, max_ts, stats_available

    def _metadata_ts_bounds(self, parquet_file: pq.ParquetFile) -> tuple[int | None, int | None]:
        try:
            ts_col = self._resolve_ts_for_parquet(self._schema_names_for_parquet(parquet_file))
        except KeyError:
            return None, None
        min_ts, max_ts, _ = self._metadata_ts_bounds_with_status(parquet_file, ts_col)
        return min_ts, max_ts

    def _scan_ts_bounds_fallback(self, file_path: Path, ts_col: str | None = None) -> tuple[int | None, int | None]:
        pf_schema = pq.read_schema(file_path)
        try:
            ts_col = ts_col or self._resolve_ts_for_parquet(pf_schema.names)
        except KeyError:
            return None, None
        table = pq.read_table(file_path, columns=[ts_col])
        if table.num_rows == 0:
            return None, None
        column = table.column(ts_col).combine_chunks()
        if column.null_count:
            column = column.drop_null()
        values = self._ts_array_to_int64(column.to_numpy(zero_copy_only=False))
        if values.size == 0:
            return None, None
        return int(values.min()), int(values.max())

    def _inspect_parquet_file(self, file_path: Path) -> dict[str, Any]:
        info: dict[str, Any] = {
            "path": str(file_path),
            "metadata_num_rows": None,
            "schema_names": [],
            "timestamp_column": None,
            "row_group_stats_available": False,
            "fallback_min_ts": None,
            "fallback_max_ts": None,
            "metadata_min_ts": None,
            "metadata_max_ts": None,
            "timestamp_status": "not_checked",
            "errors": [],
        }
        try:
            parquet_file = pq.ParquetFile(file_path)
            metadata = parquet_file.metadata
            info["metadata_num_rows"] = int(metadata.num_rows)
        except Exception as exc:
            info["timestamp_status"] = "metadata_unreadable"
            info["errors"].append(f"Metadata read error: {exc}")
            return info

        try:
            schema_names = self._schema_names_for_parquet(parquet_file)
            info["schema_names"] = schema_names
            ts_col = self._resolve_ts_for_parquet(schema_names)
            info["timestamp_column"] = ts_col
        except Exception as exc:
            info["timestamp_status"] = "missing_timestamp_column"
            info["errors"].append(f"Timestamp column not found: {exc}")
            return info

        if info["metadata_num_rows"] == 0:
            info["timestamp_status"] = "no_rows"
            return info

        try:
            min_ts, max_ts, stats_available = self._metadata_ts_bounds_with_status(parquet_file, str(info["timestamp_column"]))
            info["row_group_stats_available"] = stats_available
            info["metadata_min_ts"] = min_ts
            info["metadata_max_ts"] = max_ts
            if min_ts is not None and max_ts is not None:
                info["timestamp_status"] = "stats_available"
                return info
        except Exception as exc:
            info["errors"].append(f"Timestamp metadata stats error: {exc}")

        try:
            fmin, fmax = self._scan_ts_bounds_fallback(file_path, str(info["timestamp_column"]))
            info["fallback_min_ts"] = fmin
            info["fallback_max_ts"] = fmax
            info["timestamp_status"] = "fallback_read" if fmin is not None and fmax is not None else "fallback_empty"
        except Exception as exc:
            info["timestamp_status"] = "fallback_failed"
            info["errors"].append(f"Timestamp fallback read error: {exc}")
        return info

    def _summarize_files(self, files: list[Path]) -> tuple[int, int | None, int | None, list[dict[str, Any]]]:
        row_count = 0
        min_ts: int | None = None
        max_ts: int | None = None
        inspections: list[dict[str, Any]] = []
        for fp in files:
            info = self._inspect_parquet_file(fp)
            inspections.append(info)
            if info["metadata_num_rows"] is not None:
                row_count += int(info["metadata_num_rows"])
            fmin = info["metadata_min_ts"] if info["metadata_min_ts"] is not None else info["fallback_min_ts"]
            fmax = info["metadata_max_ts"] if info["metadata_max_ts"] is not None else info["fallback_max_ts"]
            if fmin is not None:
                min_ts = fmin if min_ts is None else min(min_ts, fmin)
            if fmax is not None:
                max_ts = fmax if max_ts is None else max(max_ts, fmax)
        return row_count, min_ts, max_ts, inspections

    def _push_gap(self, *, heap: list[tuple[int, int, int, str | None]], gap_ns: int, previous_ts: int, next_ts: int, source_file: str | None, top_n: int) -> None:
        if gap_ns <= 0:
            return
        item = (gap_ns, previous_ts, next_ts, source_file)
        if len(heap) < top_n:
            heapq.heappush(heap, item)
        elif gap_ns > heap[0][0]:
            heapq.heapreplace(heap, item)

    def _compute_gap_metrics(self, files: list[Path], *, top_n: int = 10) -> tuple[list[GapEntry], float, int]:
        heap: list[tuple[int, int, int, str | None]] = []
        prev_ts: int | None = None
        prev_file: Path | None = None
        all_ts: list[int] = []
        sbt_ns = 300 * 1_000_000_000
        sbc = 0
        for fp in files:
            fl = _relative_file_label(self.catalog_root, fp)
            try:
                pf = pq.ParquetFile(fp)
                ts_col = self._resolve_ts_for_parquet(self._schema_names_for_parquet(pf))
            except Exception:
                continue
            try:
                batches = pf.iter_batches(columns=[ts_col], batch_size=65_536)
            except Exception:
                continue
            for batch in batches:
                vals = self._ts_array_to_int64(batch.column(0).to_numpy(zero_copy_only=False))
                if vals.size == 0:
                    continue
                all_ts.extend(int(v) for v in vals.tolist())
                if prev_ts is not None:
                    tg = int(vals[0] - prev_ts)
                    if tg >= sbt_ns:
                        sbc += 1
                    ts_src = fl
                    if prev_file is not None and prev_file != fp:
                        ts_src = f"{_relative_file_label(self.catalog_root, prev_file)} -> {fl}"
                    self._push_gap(heap=heap, gap_ns=tg, previous_ts=prev_ts, next_ts=int(vals[0]), source_file=ts_src, top_n=top_n)
                if vals.size > 1:
                    diffs = np.diff(vals)
                    sbc += int(np.sum(diffs >= sbt_ns))
                    for i, g in enumerate(diffs):
                        self._push_gap(heap=heap, gap_ns=int(g), previous_ts=int(vals[i]), next_ts=int(vals[i + 1]), source_file=fl, top_n=top_n)
                prev_ts = int(vals[-1])
                prev_file = fp
        mr = estimate_missing_ratio(all_ts)
        entries = [
            GapEntry(gap_ns=gn, gap_seconds=gn / 1e9, previous_ts_event_ns=pt, previous_ts_event_iso=_ns_to_iso(pt) or "", next_ts_event_ns=nt, next_ts_event_iso=_ns_to_iso(nt) or "", source_file=sf, is_session_break=gn >= sbt_ns)
            for gn, pt, nt, sf in sorted(heap, reverse=True)
        ]
        return entries, mr, sbc

    def _scan_data_type(self, data_type: str, files: list[Path]) -> DataTypeAuditStats:
        if not files:
            return DataTypeAuditStats(data_type=data_type, present=False, status="absent", timestamp_status="not_checked", row_count_source="none")
        rc = 0; min_ts = None; max_ts = None; gaps = []; mr = 0.0; sbc = 0; errs = []; inspections: list[dict[str, Any]] = []
        try:
            rc, min_ts, max_ts, inspections = self._summarize_files(files)
        except Exception as exc:
            errs.append(f"Metadata scan error: {exc}")
        try:
            gaps, mr, sbc = self._compute_gap_metrics(files, top_n=10)
        except Exception as exc:
            errs.append(f"Gap scan error: {exc}")
        metadata_failed = bool(files) and (not inspections or any(info["metadata_num_rows"] is None for info in inspections))
        all_metadata_empty = bool(inspections) and all(info["metadata_num_rows"] == 0 for info in inspections if info["metadata_num_rows"] is not None)
        timestamp_statuses = [str(info["timestamp_status"]) for info in inspections]
        timestamp_errors = [
            error
            for info in inspections
            for error in info.get("errors", [])
            if "Timestamp" in error
        ]
        metadata_errors = [
            error
            for info in inspections
            for error in info.get("errors", [])
            if "Metadata" in error
        ]
        errs.extend(metadata_errors)
        if timestamp_errors:
            errs.extend(timestamp_errors)
        if metadata_failed:
            status = "present_unreadable"
            row_count_trusted = False
        elif rc > 0:
            status = "present_with_rows"
            row_count_trusted = True
        elif all_metadata_empty:
            status = "present_empty"
            row_count_trusted = True
        else:
            status = "present_unknown_rows"
            row_count_trusted = False
        if "stats_available" in timestamp_statuses:
            timestamp_status = "stats_available"
        elif "fallback_read" in timestamp_statuses:
            timestamp_status = "fallback_read"
        elif timestamp_statuses and all(value == "no_rows" for value in timestamp_statuses):
            timestamp_status = "no_rows"
        elif "missing_timestamp_column" in timestamp_statuses:
            timestamp_status = "missing_timestamp_column"
        elif "fallback_failed" in timestamp_statuses:
            timestamp_status = "fallback_failed"
        elif metadata_failed:
            timestamp_status = "metadata_unreadable"
        else:
            timestamp_status = "unknown"
        if not metadata_failed and rc == 0 and all_metadata_empty:
            errs = [err for err in errs if "Timestamp column not found" not in err]
        dur = (max_ts - min_ts) / 1e9 if min_ts is not None and max_ts is not None else None
        return DataTypeAuditStats(data_type=data_type, present=True, status=status, timestamp_status=timestamp_status, row_count_trusted=row_count_trusted, row_count_source="metadata" if row_count_trusted else "unknown", file_count=len(files), row_count_estimate=rc, ts_event_min_ns=min_ts, ts_event_min_iso=_ns_to_iso(min_ts), ts_event_max_ns=max_ts, ts_event_max_iso=_ns_to_iso(max_ts), duration_seconds=dur, max_gap_ns=gaps[0].gap_ns if gaps else None, max_gap_seconds=gaps[0].gap_seconds if gaps else None, missing_ratio_estimate=mr, session_break_count=sbc, top_gaps=gaps, corrupt=status == "present_unreadable", error="; ".join(errs) if errs else None)

    def _apply_convert_report_cross_checks(
        self,
        *,
        instrument_id: str,
        data_type_results: dict[str, DataTypeAuditStats],
        converter_raw: dict,
        warnings: list[CatalogWarning],
    ) -> None:
        if not converter_raw:
            return
        for data_type, stat in data_type_results.items():
            convert_rows = _converter_row_count_for_instrument(converter_raw, instrument_id, data_type)
            if convert_rows is None or convert_rows <= 0:
                continue
            audit_uncertain = (
                stat.file_count > 0
                and (
                    stat.row_count_estimate == 0
                    or not stat.row_count_trusted
                    or stat.status in {"present_empty", "present_unreadable", "present_unknown_rows"}
                )
            )
            if not audit_uncertain:
                continue
            message = (
                f"{instrument_id} {data_type}: convert report says {convert_rows} rows were written, "
                f"but audit status is {stat.status} with row_count_estimate={stat.row_count_estimate}."
            )
            warnings.append(self._warning("convert_report_audit_mismatch", message))
            stat.row_count_estimate = max(stat.row_count_estimate, convert_rows)
            stat.status = "present_unreadable"
            stat.row_count_trusted = False
            stat.row_count_source = "convert_report"
            stat.corrupt = True
            stat.error = f"{stat.error}; {message}" if stat.error else message

    def _run_l2_check(self, instrument_id: str, *, first_n: int, random_n: int, instrument_type: str, warnings: list[CatalogWarning]) -> L2CheckResult:
        cat = self._get_nautilus_catalog(warnings)
        if cat is None:
            return L2CheckResult(present=False, error="nautilus_trader unavailable; L2 sanity check skipped.")
        try:
            snaps = cat.order_book_depth10(instrument_ids=[instrument_id])
            return run_l2_checks(instrument_id=instrument_id, snapshots=snaps, instrument_type=instrument_type, first_n=first_n, random_n=random_n)
        except Exception as exc:
            return L2CheckResult(present=True, error=f"L2 sanity check failed: {exc}")

    def _compute_readiness(
        self,
        instrument_id: str,
        instrument_type: str,
        data_type_results: dict[str, DataTypeAuditStats],
        report: ReportContext,
        converter_fenced: dict | None = None,
        converter_report_found: bool = False,
    ) -> ReadinessResult:
        ts = data_type_results.get("trade_tick", DataTypeAuditStats(data_type="trade_tick"))
        ds = data_type_results.get("order_book_deltas", DataTypeAuditStats(data_type="order_book_deltas"))
        dps = data_type_results.get("order_book_depths", DataTypeAuditStats(data_type="order_book_depths"))
        ht = ts.present and ts.status != "present_empty"
        hd = ds.present and ds.status != "present_empty"
        hdp = dps.present and dps.status != "present_empty"
        dfo = hd and not hdp
        sbc = max(ts.session_break_count, ds.session_break_count)
        partial_unreadable = any(
            stat.present and stat.status in {"present_unreadable", "present_unknown_rows"} and stat.row_count_source != "convert_report"
            for stat in (ts, ds, dps)
        )

        # Fenced ranges: prefer converter report over per-instrument report file
        if converter_fenced is not None:
            fenced_count = converter_fenced["count"]
            fenced_by_reason: dict[str, int] = converter_fenced["by_reason"]
            converter_report_found = True
        elif report.report_found:
            fenced_count = len(report.fenced_ranges)
            fenced_by_reason = {}
            for fr in report.fenced_ranges:
                r = fr.reason or "unknown"
                fenced_by_reason[r] = fenced_by_reason.get(r, 0) + 1
            converter_report_found = True
        else:
            fenced_count = 0
            fenced_by_reason = {}

        lims: list[str] = []
        if not ht: lims.append("No trade_tick data")
        if not hd: lims.append("No order_book_deltas data")
        if not hdp: lims.append("No optional order_book_depths (depth10)")
        if not converter_report_found: lims.append("Converter diagnostics missing")
        if fenced_count > 0: lims.append(f"{fenced_count} fenced range(s)")
        if report.desync_count > 0: lims.append(f"{report.desync_count} desync event(s)")
        if report.resync_count > 5: lims.append(f"High resync count ({report.resync_count})")
        ic = ht or hd
        rs, breakdown = compute_readiness_breakdown(
            has_trade_tick=ht, has_order_book_deltas=hd, has_order_book_depths=hdp,
            trade_row_count=ts.row_count_estimate, delta_row_count=ds.row_count_estimate,
            depth_row_count=dps.row_count_estimate,
            trade_max_gap_seconds=ts.max_gap_seconds, delta_max_gap_seconds=ds.max_gap_seconds,
            fenced_range_count=fenced_count, desync_count=report.desync_count,
            resync_count=report.resync_count, session_break_count=sbc,
            partial_unreadable=partial_unreadable,
        )
        depth10_inspection_ready = hdp and dps.row_count_estimate > 0
        ibr = ht and hd and ts.row_count_estimate > 0 and ds.row_count_estimate > 0 and not partial_unreadable
        readiness_status = readiness_status_for_presence(
            has_trade_rows=ht and ts.row_count_estimate > 0,
            has_delta_rows=hd and ds.row_count_estimate > 0,
            has_depth_rows=hdp and dps.row_count_estimate > 0,
            partial_unreadable=partial_unreadable,
        )
        return ReadinessResult(
            instrument_id=instrument_id, instrument_type=instrument_type,
            has_trade_tick=ht, has_order_book_deltas=hd, has_order_book_depths=hdp,
            delta_first_only=dfo, is_consumable=ic, is_backtest_ready=ibr,
            depth10_inspection_ready=depth10_inspection_ready,
            trade_row_count=ts.row_count_estimate, delta_row_count=ds.row_count_estimate,
            depth_row_count=dps.row_count_estimate, trade_duration_seconds=ts.duration_seconds,
            delta_duration_seconds=ds.duration_seconds, trade_max_gap_seconds=ts.max_gap_seconds,
            delta_max_gap_seconds=ds.max_gap_seconds, session_break_count=sbc,
            fenced_range_count=fenced_count, fenced_ranges_by_reason=fenced_by_reason,
            converter_report_found=converter_report_found,
            resync_count=report.resync_count, desync_count=report.desync_count,
            snapshot_seed_count=report.snapshot_seed_count,
            backtest_readiness_score=rs,
            readiness_status=readiness_status,
            readiness_score=rs,
            score_breakdown=breakdown, limitations=lims, report=report,
        )

    def _instrument_report_signal_counts(
        self,
        instrument_id: str,
        data_type_results: dict[str, DataTypeAuditStats],
        report: ReportContext,
        converter_raw: dict,
        converter_fenced: dict | None,
    ) -> tuple[dict[str, int], list[str]]:
        fenced_count = int(converter_fenced["count"]) if converter_fenced is not None else len(report.fenced_ranges)
        desync_count = max(
            report.desync_count,
            _converter_global_count(converter_raw, "desync_events", "desync_count"),
        )
        resync_count = max(
            report.resync_count,
            _converter_global_count(converter_raw, "resync_count", "resync_events"),
        )
        bad_lines = _converter_bad_lines(converter_raw)
        missing_symbols = _converter_missing_symbol_set(converter_raw)
        partial_symbols = _converter_partial_unreadable_symbol_set(converter_raw)
        corrupt_present = sum(
            1
            for stat in data_type_results.values()
            if stat.present and (stat.corrupt or stat.row_count_estimate <= 0 or stat.ts_event_min_ns is None or stat.ts_event_max_ns is None)
        )
        missing_symbol_count = 1 if instrument_id in missing_symbols else 0
        partial_unreadable_count = corrupt_present + (1 if instrument_id in partial_symbols else 0)

        issues: list[str] = []
        if fenced_count > 0:
            issues.append(f"{fenced_count} fenced range(s)")
        if desync_count > 0:
            issues.append(f"{desync_count} desync event(s)")
        if resync_count > 0:
            issues.append(f"{resync_count} resync event(s)")
        if bad_lines > 0:
            issues.append(f"{bad_lines} bad raw line(s)")
        if missing_symbol_count > 0:
            issues.append("symbol missing in raw/converted report")
        if partial_unreadable_count > 0:
            issues.append("partial or unreadable data")

        return {
            "fenced_range_count": fenced_count,
            "desync_count": desync_count,
            "resync_count": resync_count,
            "bad_lines": bad_lines,
            "missing_symbol_count": missing_symbol_count,
            "partial_unreadable_count": partial_unreadable_count,
        }, issues

    def _compute_audit_confidence(
        self,
        *,
        data_type_results: dict[str, DataTypeAuditStats],
        converter_raw: dict,
        converter_selected_path: Path | None,
        converter_report_date: str | None,
        converter_catalog_matches: bool | None,
        converter_trade_mismatch: bool,
    ) -> tuple[float, list[str]]:
        penalty = 0.0
        issues: list[str] = []
        for data_type, stat in data_type_results.items():
            if not stat.present:
                continue
            if stat.corrupt or stat.row_count_estimate <= 0:
                penalty += 15.0
                issues.append(f"{data_type}: unreadable row count")
            if stat.ts_event_min_ns is None or stat.ts_event_max_ns is None:
                penalty += 10.0
                issues.append(f"{data_type}: null timestamp bounds")
        if converter_selected_path is None or not converter_raw:
            penalty += 15.0
            issues.append("missing convert report")
        elif converter_report_date and converter_report_date != datetime.now(tz=timezone.utc).date().isoformat() and self.convert_report_date is None:
            penalty += 5.0
            issues.append(f"convert report date {converter_report_date} is not today's UTC date")
        if converter_catalog_matches is False:
            penalty += 20.0
            issues.append("convert report catalog_root does not match scanned catalog")
        if converter_trade_mismatch:
            penalty += 15.0
            issues.append("audit/convert trade_tick count mismatch")
        return round(max(0.0, 100.0 - penalty), 2), issues

    def _build_summary(
        self,
        instrument_results: list[AuditInstrumentResult],
        *,
        converter_presence: dict | None = None,
        converter_raw: dict | None = None,
        converter_selected_path: Path | None = None,
        converter_report_date: str | None = None,
        converter_discovery_warnings: list[str] | None = None,
    ) -> AuditSummary:
        dtc = {dt: 0 for dt in EVENT_DATA_TYPES}
        trc = {dt: 0 for dt in EVENT_DATA_TYPES}
        cps: list[ChartPoint] = []
        l2ss = 0; l2bc = 0; l2bi = 0; tqs = 0; tcr = 0; tmo = 0; tem = 0; wqs = 0.0; sbc = 0
        brc = 0; cc = 0; trs = 0.0; tl2qs = 0.0; tacs = 0.0; tfr = 0; tdc = 0; trsc = 0
        for inst in instrument_results:
            for dt in EVENT_DATA_TYPES:
                st = inst.data_types.get(dt)
                if st is None: continue
                if st.present:
                    dtc[dt] += 1; trc[dt] += st.row_count_estimate
                if st.max_gap_seconds is not None:
                    cps.append(ChartPoint(instrument_id=inst.instrument_id, instrument_type=inst.instrument_type, data_type=dt, max_gap_seconds=st.max_gap_seconds))
            r = inst.readiness
            trs += r.readiness_score
            tl2qs += inst.l2_quality_score
            tacs += inst.audit_confidence_score
            if r.is_backtest_ready: brc += 1
            if r.is_consumable: cc += 1
            tfr += r.fenced_range_count; tdc += r.desync_count; trsc += r.resync_count
            l2ss += inst.l2_check.sampled_count; l2bc += inst.l2_check.bad_count
            if inst.l2_check.bad_count > 0: l2bi += 1
            tqs += inst.quality_snapshot_count; tcr += inst.crossed_count; tmo += inst.monotonic_violation_count; tem += inst.empty_side_count; sbc += inst.session_break_count
            wqs += inst.l2_quality_score * max(1, inst.quality_snapshot_count)
        l2br = l2bc / l2ss if l2ss else 0.0
        ocr = tcr / tqs if tqs else 0.0; oer = tem / tqs if tqs else 0.0; omr = tmo / tqs if tqs else 0.0
        oqs = wqs / sum(max(1, i.quality_snapshot_count) for i in instrument_results) if instrument_results else 100.0
        ars = trs / len(instrument_results) if instrument_results else 0.0
        al2qs = tl2qs / len(instrument_results) if instrument_results else 100.0
        aacs = tacs / len(instrument_results) if instrument_results else 100.0

        def _ro(inst):
            r = inst.readiness
            mg = max(r.trade_max_gap_seconds or 0, r.delta_max_gap_seconds or 0) or None
            return ReadinessOffenderItem(instrument_id=inst.instrument_id, instrument_type=inst.instrument_type, backtest_readiness_score=r.backtest_readiness_score, readiness_status=r.readiness_status, readiness_score=r.readiness_score, l2_quality_score=inst.l2_quality_score, audit_confidence_score=inst.audit_confidence_score, has_trade_tick=r.has_trade_tick, has_order_book_deltas=r.has_order_book_deltas, has_order_book_depths=r.has_order_book_depths, is_backtest_ready=r.is_backtest_ready, max_gap_seconds=mg, fenced_range_count=r.fenced_range_count, resync_count=r.resync_count, desync_count=r.desync_count, limitations=r.limitations)

        tro = [_ro(i) for i in sorted(instrument_results, key=lambda x: x.readiness.readiness_score)[:10]]
        tgo = [_ro(i) for i in sorted(instrument_results, key=lambda x: max(x.readiness.trade_max_gap_seconds or 0, x.readiness.delta_max_gap_seconds or 0), reverse=True)[:10]]
        tfo = [_ro(i) for i in sorted(instrument_results, key=lambda x: x.readiness.fenced_range_count, reverse=True)[:10]]

        def _qo(inst):
            return QualityOffenderItem(instrument_id=inst.instrument_id, instrument_type=inst.instrument_type, l2_quality_score=inst.l2_quality_score, quality_score=inst.quality_score, max_gap_seconds=inst.max_gap_seconds, crossed_rate=inst.crossed_rate, empty_rate=inst.empty_rate, bad_rate=(inst.quality_bad_snapshot_count / inst.quality_snapshot_count) if inst.quality_snapshot_count else 0.0, snapshot_count=inst.quality_snapshot_count)

        tco = [_qo(i) for i in sorted(instrument_results, key=lambda x: x.crossed_rate, reverse=True)[:10]]
        teo = [_qo(i) for i in sorted(instrument_results, key=lambda x: x.empty_rate, reverse=True)[:10]]

        viewer_tt_count = dtc.get("trade_tick", 0)
        converter_tt_count: int | None = None
        tt_mismatch = False
        tt_no_data_list: list[str] = []
        if converter_presence:
            converter_tt_count = converter_presence.get("instruments_with_trades")
            tt_no_data_list = converter_presence.get("no_trade_instrument_ids", [])
            if converter_tt_count is not None and converter_tt_count != viewer_tt_count:
                tt_mismatch = True
        raw = converter_raw or {}
        report_warnings = _converter_report_warnings(raw, converter_discovery_warnings or [])
        report_catalog_root = raw.get("catalog_root")
        report_matches = _converter_catalog_matches(raw, self.catalog_root) if raw else None
        return AuditSummary(instrument_count=len(instrument_results), data_type_coverage=dtc, total_row_counts=trc, backtest_ready_count=brc, consumable_count=cc, avg_backtest_readiness_score=round(ars, 2), avg_readiness_score=round(ars, 2), avg_l2_quality_score=round(al2qs, 2), avg_audit_confidence_score=round(aacs, 2), total_fenced_range_count=tfr, total_desync_count=tdc, total_resync_count=trsc, top_readiness_offenders=tro, top_gap_offenders=tgo, top_fenced_offenders=tfo, l2_sampled_snapshot_count=l2ss, l2_bad_count=l2bc, l2_bad_instrument_count=l2bi, l2_bad_rate=l2br, overall_crossed_rate=ocr, overall_empty_rate=oer, overall_monotonic_rate=omr, overall_quality_score=round(oqs, 2), session_break_count=sbc, chart_points=cps, top_crossed_offenders=tco, top_empty_offenders=teo, viewer_trade_tick_instrument_count=viewer_tt_count, converter_trade_tick_instrument_count=converter_tt_count, trade_tick_detected_mismatch=tt_mismatch, trade_tick_no_data_list=tt_no_data_list, convert_report_found=bool(raw and converter_selected_path), convert_report_path=str(converter_selected_path) if converter_selected_path else None, convert_report_date=converter_report_date or (str(raw.get("date")) if raw.get("date") else None), convert_report_timestamp=_converter_report_timestamp(raw), convert_report_status=str(raw.get("status")) if raw.get("status") is not None else None, convert_report_catalog_root=str(report_catalog_root) if report_catalog_root else None, convert_report_matches_catalog_root=report_matches, convert_report_paths=_converter_report_paths(raw, converter_selected_path), convert_report_warnings=report_warnings)

    def run_audit(self, *, cache_path: Path | str | None = None, first_n: int = 10, random_n: int = 10, progress_callback: ProgressCallback | None = None) -> AuditResponse:
        instrument_index, event_maps, warnings = self._collect_catalog_state()
        inventory = self.scan_inventory()
        iids = [i.instrument_id for i in inventory.instruments]
        # Load converter report and build per-instrument fenced range map.
        converter_raw, converter_selected_path, converter_report_date, converter_discovery_warnings = _selected_converter_report(
            self.converter_reports_dir,
            report_date=self.convert_report_date,
        )
        converter_fenced_map = _build_converter_fenced_map(converter_raw) if converter_raw else {}
        converter_trade_presence = _build_converter_trade_presence(converter_raw) if converter_raw else {}
        converter_matches_catalog = _converter_catalog_matches(converter_raw, self.catalog_root) if converter_raw else None
        viewer_trade_tick_count = sum(1 for iid in iids if event_maps.get("trade_tick", {}).get(iid))
        converter_trade_tick_count = converter_trade_presence.get("instruments_with_trades") if converter_trade_presence else None
        converter_trade_mismatch = converter_trade_tick_count is not None and converter_trade_tick_count != viewer_trade_tick_count
        if converter_raw and not converter_fenced_map:
            warnings.append(self._warning("converter_report_empty", "Converter report found but per_symbol_fenced_ranges is empty."))
        for warning in converter_discovery_warnings:
            warnings.append(self._warning("converter_report_warning", warning, self.converter_reports_dir))
        if converter_matches_catalog is False:
            warnings.append(self._warning("converter_catalog_root_mismatch", "Converter report catalog_root does not match the scanned catalog root.", converter_selected_path))
        total_steps = sum(1 for iid in iids for dt in EVENT_DATA_TYPES if event_maps[dt].get(iid))
        total_steps += sum(1 for iid in iids if event_maps.get("order_book_depths", {}).get(iid))
        total_steps = max(total_steps, 1)
        cs = 0
        if progress_callback: progress_callback("inventory", cs, total_steps, "Inventory scan complete, audit starting.")
        results: list[AuditInstrumentResult] = []
        for iid in iids:
            itype = instrument_index[iid]
            dtr: dict[str, DataTypeAuditStats] = {}
            for dt in EVENT_DATA_TYPES:
                files = event_maps[dt].get(iid, [])
                if files and progress_callback: progress_callback("scan", cs, total_steps, f"{iid} / {dt} scanning...")
                dtr[dt] = self._scan_data_type(dt, files)
                if files:
                    cs += 1
                    if progress_callback: progress_callback("scan", cs, total_steps, f"{iid} / {dt} done.")
            self._apply_convert_report_cross_checks(
                instrument_id=iid,
                data_type_results=dtr,
                converter_raw=converter_raw,
                warnings=warnings,
            )
            report = _load_report_context(self.report_dir, iid)
            converter_fenced = converter_fenced_map.get(iid)
            readiness = self._compute_readiness(iid, itype, dtr, report, converter_fenced=converter_fenced, converter_report_found=bool(converter_raw))
            df = event_maps.get("order_book_depths", {}).get(iid, [])
            if df and progress_callback: progress_callback("l2", cs, total_steps, f"{iid} L2 sanity check running...")
            l2r = self._run_l2_check(iid, first_n=first_n, random_n=random_n, instrument_type=itype, warnings=warnings) if df else L2CheckResult(present=False)
            fq = None
            if df:
                try: fq = self.query_service.get_l2_quality(iid)
                except Exception as exc: warnings.append(self._warning("l2_quality_failed", f"Full L2 quality aggregation failed for {iid}: {exc}"))
                cs += 1
                if progress_callback: progress_callback("l2", cs, total_steps, f"{iid} L2 sanity check done.")
            report_counts, report_quality_issues = self._instrument_report_signal_counts(
                iid,
                dtr,
                report,
                converter_raw,
                converter_fenced,
            )
            quality_snapshot_count = fq.snapshot_count if fq else l2r.sampled_count
            quality_bad_snapshot_count = fq.bad_snapshot_count if fq else l2r.bad_count
            crossed_count = fq.crossed_count if fq else l2r.crossed_count
            monotonic_violation_count = fq.monotonic_violation_count if fq else l2r.monotonic_violation_count
            empty_side_count = fq.empty_side_count if fq else l2r.empty_side_count
            max_gap_seconds = max(readiness.trade_max_gap_seconds or 0, readiness.delta_max_gap_seconds or 0) or None
            l2_quality_score = compute_l2_quality_score(
                snapshot_count=quality_snapshot_count or readiness.delta_row_count or readiness.trade_row_count,
                crossed_count=crossed_count,
                monotonic_violation_count=monotonic_violation_count,
                negative_qty_count=fq.negative_qty_count if fq else l2r.negative_qty_count,
                zero_qty_count=fq.zero_qty_count if fq else l2r.zero_qty_count,
                empty_side_count=empty_side_count,
                max_gap_seconds=max_gap_seconds,
                session_break_count=readiness.session_break_count,
                desync_count=report_counts["desync_count"],
                fenced_range_count=report_counts["fenced_range_count"],
                resync_count=report_counts["resync_count"],
                bad_lines=report_counts["bad_lines"],
                missing_symbol_count=report_counts["missing_symbol_count"],
                partial_unreadable_count=report_counts["partial_unreadable_count"],
            )
            audit_confidence_score, audit_confidence_issues = self._compute_audit_confidence(
                data_type_results=dtr,
                converter_raw=converter_raw,
                converter_selected_path=converter_selected_path,
                converter_report_date=converter_report_date,
                converter_catalog_matches=converter_matches_catalog,
                converter_trade_mismatch=converter_trade_mismatch,
            )
            l2_quality_issues = list(report_quality_issues)
            if crossed_count > 0:
                l2_quality_issues.append(f"{crossed_count} crossed book snapshot(s)")
            if monotonic_violation_count > 0:
                l2_quality_issues.append(f"{monotonic_violation_count} monotonic violation(s)")
            if empty_side_count > 0:
                l2_quality_issues.append(f"{empty_side_count} empty-side snapshot(s)")
            if readiness.session_break_count > 0:
                l2_quality_issues.append(f"{readiness.session_break_count} session break(s)")
            if max_gap_seconds is not None:
                l2_quality_issues.append(f"max gap {max_gap_seconds:.1f} seconds")
            suggestions = []
            if any(stat.file_count > 0 and stat.status in {"present_unreadable", "present_unknown_rows"} for stat in dtr.values()):
                suggestions.append("Files found but audit cannot read row counts/timestamps.")
            if readiness.session_break_count > 0:
                suggestions.append("Inspect session breaks before replaying this instrument.")
            if readiness.fenced_range_count > 0:
                suggestions.append("Review fenced ranges from the converter report.")
            if report_counts["desync_count"] > 0 or report_counts["resync_count"] > 0:
                suggestions.append("Check converter resync/desync events for book continuity.")
            results.append(AuditInstrumentResult(
                instrument_id=iid, instrument_type=itype, data_types=dtr, readiness=readiness, l2_check=l2r,
                l2_quality_score=l2_quality_score,
                data_reliability_score=l2_quality_score,
                audit_confidence_score=audit_confidence_score,
                l2_quality_issues=l2_quality_issues,
                audit_confidence_issues=audit_confidence_issues,
                reliability_suggestions=suggestions,
                quality_score=l2_quality_score,
                quality_snapshot_count=quality_snapshot_count,
                quality_bad_snapshot_count=quality_bad_snapshot_count,
                crossed_count=crossed_count,
                monotonic_violation_count=monotonic_violation_count,
                empty_side_count=empty_side_count,
                crossed_rate=fq.crossed_rate if fq else 0.0,
                empty_rate=fq.empty_side_rate if fq else 0.0,
                session_break_count=readiness.session_break_count,
                max_gap_seconds=max_gap_seconds,
                corrupt=any(s.corrupt for s in dtr.values()) or bool(fq.error if fq else False),
            ))
        summary = self._build_summary(
            results,
            converter_presence=converter_trade_presence if converter_trade_presence else None,
            converter_raw=converter_raw,
            converter_selected_path=converter_selected_path,
            converter_report_date=converter_report_date,
            converter_discovery_warnings=converter_discovery_warnings,
        )
        if summary.trade_tick_detected_mismatch:
            warnings.append(self._warning(
                "trade_tick_detected_mismatch",
                f"trade_tick mismatch: converter reports {summary.converter_trade_tick_instrument_count} instruments with trades, "
                f"viewer detected {summary.viewer_trade_tick_instrument_count}. "
                f"Instruments with no data per converter: {summary.trade_tick_no_data_list}.",
            ))
        tc = Path(cache_path).expanduser().resolve() if cache_path else self.cache_path
        audit = AuditResponse(catalog_root=str(self.catalog_root), generated_at=_utc_now_iso(), cache_path=str(tc), inventory=inventory, instruments=results, summary=summary, warnings=_dedupe_warnings(warnings + inventory.warnings))
        self.save_audit_cache(audit, tc)
        return audit

    def debug_trade_tick(self) -> dict:
        """Return a diagnostic snapshot of the TradeTick data found in the catalog.

        Includes:
        - discovered dataset paths
        - number of parquet files per instrument
        - sample schema (first file of first instrument)
        - first 5 instrument IDs found
        - whether any futures/perpetual instrument IDs are present
        """
        trade_tick_dir = self.data_root / "trade_tick"
        result: dict = {
            "trade_tick_dir": str(trade_tick_dir),
            "trade_tick_dir_exists": trade_tick_dir.exists(),
            "instrument_count": 0,
            "total_parquet_files": 0,
            "instruments": [],
            "sample_schema": None,
            "futures_instruments_found": [],
            "spot_instruments_found": [],
        }
        if not trade_tick_dir.exists():
            result["error"] = f"Directory not found: {trade_tick_dir}"
            return result

        instruments: list[dict] = []
        for instrument_dir in sorted(trade_tick_dir.iterdir()):
            if not instrument_dir.is_dir():
                continue
            parquet_files = sorted(instrument_dir.glob("*.parquet"))
            file_count = len(parquet_files)
            if file_count == 0:
                continue
            entry: dict = {
                "instrument_id": instrument_dir.name,
                "file_count": file_count,
                "files": [str(p) for p in parquet_files[:3]],
                "schema": None,
                "row_count": None,
                "ts_event_min_ns": None,
                "ts_event_max_ns": None,
                "error": None,
            }
            try:
                schema = pq.read_schema(parquet_files[0])
                entry["schema"] = schema.names
                pf = pq.ParquetFile(parquet_files[0])
                entry["row_count"] = pf.metadata.num_rows
                fmin, fmax = self._metadata_ts_bounds(pf)
                if fmin is None or fmax is None:
                    fmin, fmax = self._scan_ts_bounds_fallback(parquet_files[0])
                entry["ts_event_min_ns"] = fmin
                entry["ts_event_max_ns"] = fmax
            except Exception as exc:
                entry["error"] = str(exc)
            instruments.append(entry)

        result["instrument_count"] = len(instruments)
        result["total_parquet_files"] = sum(e["file_count"] for e in instruments)
        result["instruments"] = instruments[:5]  # first 5 for debug output

        if instruments:
            result["sample_schema"] = instruments[0].get("schema")

        result["futures_instruments_found"] = [
            e["instrument_id"] for e in instruments if "-PERP." in e["instrument_id"] or "-SWAP." in e["instrument_id"]
        ]
        result["spot_instruments_found"] = [
            e["instrument_id"] for e in instruments if "-PERP." not in e["instrument_id"] and "-SWAP." not in e["instrument_id"]
        ]
        return result

    def debug_parquet(self, *, instrument_id: str, data_type: str) -> dict:
        """Return per-file parquet metadata/timestamp interpretation for one dataset."""
        warnings: list[CatalogWarning] = []
        files = self._event_file_map(data_type, warnings).get(instrument_id, [])
        stats = self._scan_data_type(data_type, files)
        per_file = []
        for file_path in files:
            info = self._inspect_parquet_file(file_path)
            info["path"] = str(file_path)
            info["relative_path"] = _relative_file_label(self.catalog_root, file_path)
            per_file.append(info)
        return {
            "catalog_root": str(self.catalog_root),
            "instrument_id": instrument_id,
            "data_type": data_type,
            "resolved_file_paths": [str(path) for path in files],
            "files": per_file,
            "final_status": stats.status,
            "timestamp_status": stats.timestamp_status,
            "row_count_estimate": stats.row_count_estimate,
            "row_count_trusted": stats.row_count_trusted,
            "row_count_source": stats.row_count_source,
            "ts_event_min_ns": stats.ts_event_min_ns,
            "ts_event_max_ns": stats.ts_event_max_ns,
            "error": stats.error,
            "warnings": [warning.model_dump(mode="json") for warning in warnings],
        }

    def save_audit_cache(self, audit: AuditResponse, cache_path: Path | str | None = None) -> Path:
        target = Path(cache_path).expanduser().resolve() if cache_path else self.cache_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(audit.model_dump_json(indent=2), encoding="utf-8")
        return target

    def load_audit_cache(self, cache_path: Path | str | None = None) -> AuditResponse | None:
        target = Path(cache_path).expanduser().resolve() if cache_path else self.cache_path
        if not target.exists():
            return None
        return AuditResponse.model_validate_json(target.read_text(encoding="utf-8"))
