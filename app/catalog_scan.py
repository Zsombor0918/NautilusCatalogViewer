from __future__ import annotations

import heapq
import json
import math
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pyarrow.parquet as pq

from .l2_checks import estimate_missing_ratio, run_l2_checks
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
    score = 0.0
    if has_trade_tick:
        score += 20.0
    if has_order_book_deltas:
        score += 25.0
    if has_order_book_depths:
        score += 5.0
    if trade_row_count > 1000:
        score += 10.0
    elif trade_row_count > 0:
        score += 5.0
    if delta_row_count > 1000:
        score += 10.0
    elif delta_row_count > 0:
        score += 5.0
    gap_penalty = 0.0
    for gap_sec in (trade_max_gap_seconds, delta_max_gap_seconds):
        if gap_sec is not None and gap_sec > 0:
            gap_penalty += min(7.5, math.log10(gap_sec + 1.0) * 2.5)
    score -= gap_penalty
    score -= min(10.0, fenced_range_count * 2.0)
    score -= min(5.0, desync_count * 1.0)
    score -= min(5.0, resync_count * 0.5)
    score -= min(5.0, session_break_count * 0.5)
    return round(max(0.0, min(100.0, score)), 2)


def _converter_key_to_instrument_id(key: str) -> str:
    """Map 'BINANCE_SPOT/AAVEUSDT' -> 'AAVEUSDT.BINANCE' and 'BINANCE_USDTF/AAVEUSDT' -> 'AAVEUSDT-PERP.BINANCE'."""
    parts = key.split("/", 1)
    if len(parts) != 2:
        return key
    venue, symbol = parts
    exchange = venue.split("_")[0]  # e.g. BINANCE from BINANCE_SPOT
    is_perp = any(tag in venue for tag in ("USDTF", "PERP", "FUTURES", "SWAP"))
    return f"{symbol}-PERP.{exchange}" if is_perp else f"{symbol}.{exchange}"


def _load_converter_report(converter_reports_dir: Path) -> dict:
    """Load the latest YYYY-MM-DD.json from the converter reports directory.

    Returns the parsed JSON dict or an empty dict if none found.
    """
    if not converter_reports_dir.exists():
        return {}
    json_files = sorted(converter_reports_dir.glob("*.json"))
    if not json_files:
        return {}
    latest = json_files[-1]  # alphabetical == date order
    try:
        return json.loads(latest.read_text(encoding="utf-8"))
    except Exception:
        return {}


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
    ) -> None:
        self.catalog_root = Path(catalog_root).expanduser().resolve()
        project_root = Path(__file__).resolve().parent.parent
        default_cache = project_root / "state" / "web_audit_cache.json"
        self.cache_path = Path(cache_path).expanduser().resolve() if cache_path else default_cache
        # Converter reports directory: explicit arg > env var > None
        import os as _os
        _env = _os.getenv("NAUTILUS_CONVERTER_REPORTS_DIR")
        if converter_reports_dir is not None:
            self.converter_reports_dir: Path | None = Path(converter_reports_dir).expanduser().resolve()
        elif _env:
            self.converter_reports_dir = Path(_env).expanduser().resolve()
        else:
            self.converter_reports_dir = None
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
            warnings.append(self._warning("missing_catalog_root", "Catalog root does not exist.", self.catalog_root))
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

    def _metadata_ts_bounds(self, parquet_file: pq.ParquetFile) -> tuple[int | None, int | None]:
        metadata = parquet_file.metadata
        min_ts: int | None = None
        max_ts: int | None = None
        for rgi in range(metadata.num_row_groups):
            rg = metadata.row_group(rgi)
            for ci in range(rg.num_columns):
                col = rg.column(ci)
                if col.path_in_schema != "ts_event" or col.statistics is None:
                    continue
                stats = col.statistics
                cmin, cmax = int(stats.min), int(stats.max)
                min_ts = cmin if min_ts is None else min(min_ts, cmin)
                max_ts = cmax if max_ts is None else max(max_ts, cmax)
        return min_ts, max_ts

    def _scan_ts_bounds_fallback(self, file_path: Path) -> tuple[int | None, int | None]:
        table = pq.read_table(file_path, columns=["ts_event"])
        if table.num_rows == 0:
            return None, None
        values = np.asarray(table.column("ts_event").combine_chunks().to_numpy(), dtype=np.int64)
        if values.size == 0:
            return None, None
        return int(values.min()), int(values.max())

    def _summarize_files(self, files: list[Path]) -> tuple[int, int | None, int | None]:
        row_count = 0
        min_ts: int | None = None
        max_ts: int | None = None
        for fp in files:
            pf = pq.ParquetFile(fp)
            row_count += pf.metadata.num_rows
            fmin, fmax = self._metadata_ts_bounds(pf)
            if fmin is None or fmax is None:
                fmin, fmax = self._scan_ts_bounds_fallback(fp)
            if fmin is not None:
                min_ts = fmin if min_ts is None else min(min_ts, fmin)
            if fmax is not None:
                max_ts = fmax if max_ts is None else max(max_ts, fmax)
        return row_count, min_ts, max_ts

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
            pf = pq.ParquetFile(fp)
            fl = _relative_file_label(self.catalog_root, fp)
            for batch in pf.iter_batches(columns=["ts_event"], batch_size=65_536):
                vals = np.asarray(batch.column(0).to_numpy(), dtype=np.int64)
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
            return DataTypeAuditStats(data_type=data_type, present=False)
        rc = 0; min_ts = None; max_ts = None; gaps = []; mr = 0.0; sbc = 0; errs = []
        try:
            rc, min_ts, max_ts = self._summarize_files(files)
        except Exception as exc:
            errs.append(f"Metadata scan error: {exc}")
        try:
            gaps, mr, sbc = self._compute_gap_metrics(files, top_n=10)
        except Exception as exc:
            errs.append(f"Gap scan error: {exc}")
        dur = (max_ts - min_ts) / 1e9 if min_ts is not None and max_ts is not None else None
        return DataTypeAuditStats(data_type=data_type, present=True, file_count=len(files), row_count_estimate=rc, ts_event_min_ns=min_ts, ts_event_min_iso=_ns_to_iso(min_ts), ts_event_max_ns=max_ts, ts_event_max_iso=_ns_to_iso(max_ts), duration_seconds=dur, max_gap_ns=gaps[0].gap_ns if gaps else None, max_gap_seconds=gaps[0].gap_seconds if gaps else None, missing_ratio_estimate=mr, session_break_count=sbc, top_gaps=gaps, corrupt=bool(errs), error="; ".join(errs) if errs else None)

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
    ) -> ReadinessResult:
        ts = data_type_results.get("trade_tick", DataTypeAuditStats(data_type="trade_tick"))
        ds = data_type_results.get("order_book_deltas", DataTypeAuditStats(data_type="order_book_deltas"))
        dps = data_type_results.get("order_book_depths", DataTypeAuditStats(data_type="order_book_depths"))
        ht, hd, hdp = ts.present, ds.present, dps.present
        dfo = hd and not hdp
        sbc = max(ts.session_break_count, ds.session_break_count)

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
            converter_report_found = False

        lims: list[str] = []
        if not ht: lims.append("No trade_tick data")
        if not hd: lims.append("No order_book_deltas data")
        if not hdp: lims.append("No optional order_book_depths (depth10)")
        if not converter_report_found: lims.append("Converter diagnostics missing")
        if fenced_count > 0: lims.append(f"{fenced_count} fenced range(s)")
        if report.desync_count > 0: lims.append(f"{report.desync_count} desync event(s)")
        if report.resync_count > 5: lims.append(f"High resync count ({report.resync_count})")
        ic = ht or hd
        ibr = ht and hd and report.desync_count == 0
        # Slight penalty when converter diagnostics are missing
        missing_report_penalty = 0 if converter_report_found else 5
        rs = max(0.0, compute_readiness_score(
            has_trade_tick=ht, has_order_book_deltas=hd, has_order_book_depths=hdp,
            trade_row_count=ts.row_count_estimate, delta_row_count=ds.row_count_estimate,
            trade_max_gap_seconds=ts.max_gap_seconds, delta_max_gap_seconds=ds.max_gap_seconds,
            fenced_range_count=fenced_count, desync_count=report.desync_count,
            resync_count=report.resync_count, session_break_count=sbc,
        ) - missing_report_penalty)
        return ReadinessResult(
            instrument_id=instrument_id, instrument_type=instrument_type,
            has_trade_tick=ht, has_order_book_deltas=hd, has_order_book_depths=hdp,
            delta_first_only=dfo, is_consumable=ic, is_backtest_ready=ibr,
            trade_row_count=ts.row_count_estimate, delta_row_count=ds.row_count_estimate,
            depth_row_count=dps.row_count_estimate, trade_duration_seconds=ts.duration_seconds,
            delta_duration_seconds=ds.duration_seconds, trade_max_gap_seconds=ts.max_gap_seconds,
            delta_max_gap_seconds=ds.max_gap_seconds, session_break_count=sbc,
            fenced_range_count=fenced_count, fenced_ranges_by_reason=fenced_by_reason,
            converter_report_found=converter_report_found,
            resync_count=report.resync_count, desync_count=report.desync_count,
            snapshot_seed_count=report.snapshot_seed_count,
            readiness_score=rs, limitations=lims, report=report,
        )

    def _build_summary(self, instrument_results: list[AuditInstrumentResult]) -> AuditSummary:
        dtc = {dt: 0 for dt in EVENT_DATA_TYPES}
        trc = {dt: 0 for dt in EVENT_DATA_TYPES}
        cps: list[ChartPoint] = []
        l2ss = 0; l2bc = 0; l2bi = 0; tqs = 0; tcr = 0; tmo = 0; tem = 0; wqs = 0.0; sbc = 0
        brc = 0; cc = 0; trs = 0.0; tfr = 0; tdc = 0; trsc = 0
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
            if r.is_backtest_ready: brc += 1
            if r.is_consumable: cc += 1
            tfr += r.fenced_range_count; tdc += r.desync_count; trsc += r.resync_count
            l2ss += inst.l2_check.sampled_count; l2bc += inst.l2_check.bad_count
            if inst.l2_check.bad_count > 0: l2bi += 1
            tqs += inst.quality_snapshot_count; tcr += inst.crossed_count; tmo += inst.monotonic_violation_count; tem += inst.empty_side_count; sbc += inst.session_break_count
            wqs += inst.quality_score * max(1, inst.quality_snapshot_count)
        l2br = l2bc / l2ss if l2ss else 0.0
        ocr = tcr / tqs if tqs else 0.0; oer = tem / tqs if tqs else 0.0; omr = tmo / tqs if tqs else 0.0
        oqs = wqs / sum(max(1, i.quality_snapshot_count) for i in instrument_results) if instrument_results else 100.0
        ars = trs / len(instrument_results) if instrument_results else 0.0

        def _ro(inst):
            r = inst.readiness
            mg = max(r.trade_max_gap_seconds or 0, r.delta_max_gap_seconds or 0) or None
            return ReadinessOffenderItem(instrument_id=inst.instrument_id, instrument_type=inst.instrument_type, readiness_score=r.readiness_score, has_trade_tick=r.has_trade_tick, has_order_book_deltas=r.has_order_book_deltas, has_order_book_depths=r.has_order_book_depths, is_backtest_ready=r.is_backtest_ready, max_gap_seconds=mg, fenced_range_count=r.fenced_range_count, resync_count=r.resync_count, desync_count=r.desync_count, limitations=r.limitations)

        tro = [_ro(i) for i in sorted(instrument_results, key=lambda x: x.readiness.readiness_score)[:10]]
        tgo = [_ro(i) for i in sorted(instrument_results, key=lambda x: max(x.readiness.trade_max_gap_seconds or 0, x.readiness.delta_max_gap_seconds or 0), reverse=True)[:10]]
        tfo = [_ro(i) for i in sorted(instrument_results, key=lambda x: x.readiness.fenced_range_count, reverse=True)[:10]]

        def _qo(inst):
            return QualityOffenderItem(instrument_id=inst.instrument_id, instrument_type=inst.instrument_type, quality_score=inst.quality_score, max_gap_seconds=inst.max_gap_seconds, crossed_rate=inst.crossed_rate, empty_rate=inst.empty_rate, bad_rate=(inst.quality_bad_snapshot_count / inst.quality_snapshot_count) if inst.quality_snapshot_count else 0.0, snapshot_count=inst.quality_snapshot_count)

        tco = [_qo(i) for i in sorted(instrument_results, key=lambda x: x.crossed_rate, reverse=True)[:10]]
        teo = [_qo(i) for i in sorted(instrument_results, key=lambda x: x.empty_rate, reverse=True)[:10]]

        return AuditSummary(instrument_count=len(instrument_results), data_type_coverage=dtc, total_row_counts=trc, backtest_ready_count=brc, consumable_count=cc, avg_readiness_score=round(ars, 2), total_fenced_range_count=tfr, total_desync_count=tdc, total_resync_count=trsc, top_readiness_offenders=tro, top_gap_offenders=tgo, top_fenced_offenders=tfo, l2_sampled_snapshot_count=l2ss, l2_bad_count=l2bc, l2_bad_instrument_count=l2bi, l2_bad_rate=l2br, overall_crossed_rate=ocr, overall_empty_rate=oer, overall_monotonic_rate=omr, overall_quality_score=oqs, session_break_count=sbc, chart_points=cps, top_crossed_offenders=tco, top_empty_offenders=teo)

    def run_audit(self, *, cache_path: Path | str | None = None, first_n: int = 10, random_n: int = 10, progress_callback: ProgressCallback | None = None) -> AuditResponse:
        instrument_index, event_maps, warnings = self._collect_catalog_state()
        inventory = self.scan_inventory()
        iids = [i.instrument_id for i in inventory.instruments]
        # Load converter report and build per-instrument fenced range map
        converter_raw = _load_converter_report(self.converter_reports_dir) if self.converter_reports_dir else {}
        converter_fenced_map = _build_converter_fenced_map(converter_raw) if converter_raw else {}
        if converter_raw and not converter_fenced_map:
            warnings.append(self._warning("converter_report_empty", "Converter report found but per_symbol_fenced_ranges is empty."))
        if self.converter_reports_dir and not converter_raw:
            warnings.append(self._warning("converter_report_missing", f"No converter reports found in {self.converter_reports_dir}", self.converter_reports_dir))
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
            report = _load_report_context(self.report_dir, iid)
            converter_fenced = converter_fenced_map.get(iid)
            readiness = self._compute_readiness(iid, itype, dtr, report, converter_fenced=converter_fenced)
            df = event_maps.get("order_book_depths", {}).get(iid, [])
            if df and progress_callback: progress_callback("l2", cs, total_steps, f"{iid} L2 sanity check running...")
            l2r = self._run_l2_check(iid, first_n=first_n, random_n=random_n, instrument_type=itype, warnings=warnings) if df else L2CheckResult(present=False)
            fq = None
            if df:
                try: fq = self.query_service.get_l2_quality(iid)
                except Exception as exc: warnings.append(self._warning("l2_quality_failed", f"Full L2 quality aggregation failed for {iid}: {exc}"))
                cs += 1
                if progress_callback: progress_callback("l2", cs, total_steps, f"{iid} L2 sanity check done.")
            results.append(AuditInstrumentResult(
                instrument_id=iid, instrument_type=itype, data_types=dtr, readiness=readiness, l2_check=l2r,
                quality_score=fq.quality_score if fq else l2r.quality_score,
                quality_snapshot_count=fq.snapshot_count if fq else l2r.sampled_count,
                quality_bad_snapshot_count=fq.bad_snapshot_count if fq else l2r.bad_count,
                crossed_count=fq.crossed_count if fq else l2r.crossed_count,
                monotonic_violation_count=fq.monotonic_violation_count if fq else l2r.monotonic_violation_count,
                empty_side_count=fq.empty_side_count if fq else l2r.empty_side_count,
                crossed_rate=fq.crossed_rate if fq else 0.0,
                empty_rate=fq.empty_side_rate if fq else 0.0,
                session_break_count=readiness.session_break_count,
                max_gap_seconds=max(readiness.trade_max_gap_seconds or 0, readiness.delta_max_gap_seconds or 0) or None,
                corrupt=any(s.corrupt for s in dtr.values()) or bool(fq.error if fq else False),
            ))
        summary = self._build_summary(results)
        tc = Path(cache_path).expanduser().resolve() if cache_path else self.cache_path
        audit = AuditResponse(catalog_root=str(self.catalog_root), generated_at=_utc_now_iso(), cache_path=str(tc), inventory=inventory, instruments=results, summary=summary, warnings=_dedupe_warnings(warnings + inventory.warnings))
        self.save_audit_cache(audit, tc)
        return audit

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
