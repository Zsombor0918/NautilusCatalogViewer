from __future__ import annotations

import heapq
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
    GapEntry,
    InstrumentCoverage,
    InstrumentInventoryItem,
    InstrumentTypeSummary,
    InventoryResponse,
    L2CheckResult,
    QualityOffenderItem,
)
from .query import CatalogQueryService

try:
    from nautilus_trader.persistence.catalog import ParquetDataCatalog
except Exception:  # pragma: no cover - exercised only when Nautilus is absent.
    ParquetDataCatalog = None  # type: ignore[assignment]


INSTRUMENT_TYPE_DIRS: tuple[str, ...] = ("crypto_perpetual", "currency_pair")
EVENT_DATA_TYPES: tuple[str, ...] = ("trade_tick", "order_book_depths")

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


class CatalogScanner:
    def __init__(self, catalog_root: Path | str, cache_path: Path | str | None = None) -> None:
        self.catalog_root = Path(catalog_root).expanduser().resolve()
        project_root = Path(__file__).resolve().parent.parent
        default_cache = project_root / "state" / "web_audit_cache.json"
        self.cache_path = Path(cache_path).expanduser().resolve() if cache_path else default_cache
        self._nautilus_catalog: Any | None = None
        self._nautilus_catalog_initialized = False
        self.query_service = CatalogQueryService(self.catalog_root)

    @property
    def data_root(self) -> Path:
        return self.catalog_root / "data"

    def _warning(self, code: str, message: str, path: Path | None = None) -> CatalogWarning:
        return CatalogWarning(code=code, message=message, path=str(path) if path else None)

    def _get_nautilus_catalog(self, warnings: list[CatalogWarning]) -> Any | None:
        if self._nautilus_catalog_initialized:
            return self._nautilus_catalog

        self._nautilus_catalog_initialized = True
        if ParquetDataCatalog is None:
            warnings.append(
                self._warning(
                    "nautilus_unavailable",
                    "nautilus_trader nincs telepítve, ezért a catalog listázás filesystem fallbackre vált.",
                ),
            )
            return None

        try:
            self._nautilus_catalog = ParquetDataCatalog(str(self.catalog_root))
        except Exception as exc:
            warnings.append(
                self._warning(
                    "nautilus_catalog_init_failed",
                    f"Nem sikerült megnyitni a ParquetDataCatalogot: {exc}",
                    self.catalog_root,
                ),
            )
            self._nautilus_catalog = None
        return self._nautilus_catalog

    def _load_instrument_index(self, warnings: list[CatalogWarning]) -> dict[str, str]:
        instrument_index: dict[str, str] = {}

        for instrument_type in INSTRUMENT_TYPE_DIRS:
            type_dir = self.data_root / instrument_type
            if not type_dir.exists():
                warnings.append(
                    self._warning(
                        "missing_instrument_type_dir",
                        f"Hiányzik az instrument type mappa: {instrument_type}",
                        type_dir,
                    ),
                )
                continue

            for instrument_dir in sorted(path for path in type_dir.iterdir() if path.is_dir()):
                instrument_index[instrument_dir.name] = instrument_type

        nautilus_catalog = self._get_nautilus_catalog(warnings)
        if nautilus_catalog is not None:
            try:
                for instrument in nautilus_catalog.instruments():
                    instrument_index[instrument.id.value] = _snake_case(type(instrument).__name__)
            except Exception as exc:
                warnings.append(
                    self._warning(
                        "nautilus_instrument_list_failed",
                        f"A Nautilus instrument lista nem olvasható: {exc}",
                        self.catalog_root,
                    ),
                )

        return instrument_index

    def _event_file_map(self, data_type: str, warnings: list[CatalogWarning]) -> dict[str, list[Path]]:
        type_dir = self.data_root / data_type
        file_map: dict[str, list[Path]] = {}

        if not type_dir.exists():
            warnings.append(
                self._warning(
                    "missing_data_type_dir",
                    f"Hiányzik az adat mappa: {data_type}",
                    type_dir,
                ),
            )
            return file_map

        for instrument_dir in sorted(path for path in type_dir.iterdir() if path.is_dir()):
            parquet_files = sorted(path for path in instrument_dir.glob("*.parquet") if path.is_file())
            file_map[instrument_dir.name] = parquet_files

        return file_map

    def _infer_instrument_type(self, instrument_id: str) -> str:
        if "-PERP." in instrument_id:
            return "crypto_perpetual"
        return "currency_pair"

    def _collect_catalog_state(
        self,
    ) -> tuple[dict[str, str], dict[str, dict[str, list[Path]]], list[CatalogWarning]]:
        warnings: list[CatalogWarning] = []

        if not self.catalog_root.exists():
            warnings.append(
                self._warning(
                    "missing_catalog_root",
                    "A catalog root nem létezik.",
                    self.catalog_root,
                ),
            )

        if not self.data_root.exists():
            warnings.append(
                self._warning(
                    "missing_data_root",
                    "A catalog data mappa nem létezik.",
                    self.data_root,
                ),
            )

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

        for instrument_id in sorted(
            instrument_index,
            key=lambda value: (instrument_index[value], value),
        ):
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
            instruments.append(
                InstrumentInventoryItem(
                    instrument_id=instrument_id,
                    instrument_type=instrument_type,
                    has_any_data=has_any_data,
                    coverage=coverage,
                ),
            )

        instrument_types = [
            InstrumentTypeSummary(
                instrument_type=instrument_type,
                instrument_count=len(names),
                with_any_data_count=grouped_with_data[instrument_type],
                instruments=names,
            )
            for instrument_type, names in sorted(grouped.items())
        ]

        return InventoryResponse(
            catalog_root=str(self.catalog_root),
            generated_at=_utc_now_iso(),
            available_data_types=list(EVENT_DATA_TYPES),
            instrument_types=instrument_types,
            instruments=instruments,
            warnings=warnings,
        )

    def _metadata_ts_bounds(self, parquet_file: pq.ParquetFile) -> tuple[int | None, int | None]:
        metadata = parquet_file.metadata
        min_ts: int | None = None
        max_ts: int | None = None

        for row_group_index in range(metadata.num_row_groups):
            row_group = metadata.row_group(row_group_index)
            for column_index in range(row_group.num_columns):
                column = row_group.column(column_index)
                if column.path_in_schema != "ts_event" or column.statistics is None:
                    continue

                stats = column.statistics
                current_min = int(stats.min)
                current_max = int(stats.max)
                min_ts = current_min if min_ts is None else min(min_ts, current_min)
                max_ts = current_max if max_ts is None else max(max_ts, current_max)
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

        for file_path in files:
            parquet_file = pq.ParquetFile(file_path)
            row_count += parquet_file.metadata.num_rows
            file_min, file_max = self._metadata_ts_bounds(parquet_file)
            if file_min is None or file_max is None:
                file_min, file_max = self._scan_ts_bounds_fallback(file_path)

            if file_min is not None:
                min_ts = file_min if min_ts is None else min(min_ts, file_min)
            if file_max is not None:
                max_ts = file_max if max_ts is None else max(max_ts, file_max)

        return row_count, min_ts, max_ts

    def _push_gap(
        self,
        *,
        heap: list[tuple[int, int, int, str | None]],
        gap_ns: int,
        previous_ts: int,
        next_ts: int,
        source_file: str | None,
        top_n: int,
    ) -> None:
        if gap_ns <= 0:
            return

        item = (gap_ns, previous_ts, next_ts, source_file)
        if len(heap) < top_n:
            heapq.heappush(heap, item)
            return

        if gap_ns > heap[0][0]:
            heapq.heapreplace(heap, item)

    def _compute_gap_metrics(self, files: list[Path], *, top_n: int = 10) -> tuple[list[GapEntry], float, int]:
        heap: list[tuple[int, int, int, str | None]] = []
        previous_ts: int | None = None
        previous_file: Path | None = None
        all_ts_values: list[int] = []
        session_break_threshold_ns = 300 * 1_000_000_000
        session_break_count = 0

        for file_path in files:
            parquet_file = pq.ParquetFile(file_path)
            file_label = _relative_file_label(self.catalog_root, file_path)

            for batch in parquet_file.iter_batches(columns=["ts_event"], batch_size=65_536):
                values = np.asarray(batch.column(0).to_numpy(), dtype=np.int64)
                if values.size == 0:
                    continue
                all_ts_values.extend(int(value) for value in values.tolist())

                if previous_ts is not None:
                    transition_gap = int(values[0] - previous_ts)
                    if transition_gap >= session_break_threshold_ns:
                        session_break_count += 1
                    transition_source = file_label
                    if previous_file is not None and previous_file != file_path:
                        previous_label = _relative_file_label(self.catalog_root, previous_file)
                        transition_source = f"{previous_label} -> {file_label}"
                    self._push_gap(
                        heap=heap,
                        gap_ns=transition_gap,
                        previous_ts=previous_ts,
                        next_ts=int(values[0]),
                        source_file=transition_source,
                        top_n=top_n,
                    )

                if values.size > 1:
                    diffs = np.diff(values)
                    session_break_count += int(np.sum(diffs >= session_break_threshold_ns))
                    for index, gap in enumerate(diffs):
                        self._push_gap(
                            heap=heap,
                            gap_ns=int(gap),
                            previous_ts=int(values[index]),
                            next_ts=int(values[index + 1]),
                            source_file=file_label,
                            top_n=top_n,
                        )

                previous_ts = int(values[-1])
                previous_file = file_path

        missing_ratio_estimate = estimate_missing_ratio(all_ts_values)
        entries = [
            GapEntry(
                gap_ns=gap_ns,
                gap_seconds=gap_ns / 1_000_000_000,
                previous_ts_event_ns=prev_ts,
                previous_ts_event_iso=_ns_to_iso(prev_ts) or "",
                next_ts_event_ns=next_ts,
                next_ts_event_iso=_ns_to_iso(next_ts) or "",
                source_file=source_file,
                is_session_break=gap_ns >= session_break_threshold_ns,
            )
            for gap_ns, prev_ts, next_ts, source_file in sorted(heap, reverse=True)
        ]
        return entries, missing_ratio_estimate, session_break_count

    def _scan_data_type(self, data_type: str, files: list[Path]) -> DataTypeAuditStats:
        if not files:
            return DataTypeAuditStats(data_type=data_type, present=False)

        row_count = 0
        min_ts: int | None = None
        max_ts: int | None = None
        gaps: list[GapEntry] = []
        missing_ratio_estimate = 0.0
        session_break_count = 0
        errors: list[str] = []

        try:
            row_count, min_ts, max_ts = self._summarize_files(files)
        except Exception as exc:
            errors.append(f"Metadata scan hiba: {exc}")

        try:
            gaps, missing_ratio_estimate, session_break_count = self._compute_gap_metrics(files, top_n=10)
        except Exception as exc:
            errors.append(f"Gap scan hiba: {exc}")

        duration_seconds = None
        if min_ts is not None and max_ts is not None:
            duration_seconds = (max_ts - min_ts) / 1_000_000_000

        max_gap_ns = gaps[0].gap_ns if gaps else None
        max_gap_seconds = gaps[0].gap_seconds if gaps else None

        return DataTypeAuditStats(
            data_type=data_type,
            present=True,
            file_count=len(files),
            row_count_estimate=row_count,
            ts_event_min_ns=min_ts,
            ts_event_min_iso=_ns_to_iso(min_ts),
            ts_event_max_ns=max_ts,
            ts_event_max_iso=_ns_to_iso(max_ts),
            duration_seconds=duration_seconds,
            max_gap_ns=max_gap_ns,
            max_gap_seconds=max_gap_seconds,
            missing_ratio_estimate=missing_ratio_estimate,
            session_break_count=session_break_count,
            top_gaps=gaps,
            corrupt=bool(errors),
            error="; ".join(errors) if errors else None,
        )

    def _run_l2_check(
        self,
        instrument_id: str,
        *,
        first_n: int,
        random_n: int,
        instrument_type: str,
        warnings: list[CatalogWarning],
    ) -> L2CheckResult:
        nautilus_catalog = self._get_nautilus_catalog(warnings)
        if nautilus_catalog is None:
            return L2CheckResult(
                present=False,
                error="A nautilus_trader nem érhető el, ezért az L2 sanity check nem futott le.",
            )

        try:
            snapshots = nautilus_catalog.order_book_depth10(instrument_ids=[instrument_id])
            return run_l2_checks(
                instrument_id=instrument_id,
                snapshots=snapshots,
                instrument_type=instrument_type,
                first_n=first_n,
                random_n=random_n,
            )
        except Exception as exc:
            return L2CheckResult(
                present=True,
                error=f"Az L2 sanity check nem sikerült: {exc}",
            )

    def _build_summary(self, instrument_results: list[AuditInstrumentResult]) -> AuditSummary:
        data_type_coverage = {data_type: 0 for data_type in EVENT_DATA_TYPES}
        total_row_counts = {data_type: 0 for data_type in EVENT_DATA_TYPES}
        chart_points: list[ChartPoint] = []
        l2_sampled_snapshot_count = 0
        l2_bad_count = 0
        l2_bad_instrument_count = 0
        total_quality_snapshots = 0
        total_crossed = 0
        total_monotonic = 0
        total_empty = 0
        weighted_quality_score = 0.0
        session_break_count = 0

        for instrument in instrument_results:
            for data_type in EVENT_DATA_TYPES:
                stats = instrument.data_types[data_type]
                if stats.present:
                    data_type_coverage[data_type] += 1
                    total_row_counts[data_type] += stats.row_count_estimate
                if stats.max_gap_seconds is not None:
                    chart_points.append(
                        ChartPoint(
                            instrument_id=instrument.instrument_id,
                            instrument_type=instrument.instrument_type,
                            data_type=data_type,
                            max_gap_seconds=stats.max_gap_seconds,
                        ),
                    )

            l2_sampled_snapshot_count += instrument.l2_check.sampled_count
            l2_bad_count += instrument.l2_check.bad_count
            if instrument.l2_check.bad_count > 0:
                l2_bad_instrument_count += 1
            total_quality_snapshots += instrument.quality_snapshot_count
            total_crossed += instrument.crossed_count
            total_monotonic += instrument.monotonic_violation_count
            total_empty += instrument.empty_side_count
            session_break_count += instrument.session_break_count
            weighted_quality_score += instrument.quality_score * max(1, instrument.quality_snapshot_count)

        l2_bad_rate = (
            l2_bad_count / l2_sampled_snapshot_count
            if l2_sampled_snapshot_count
            else 0.0
        )
        overall_crossed_rate = total_crossed / total_quality_snapshots if total_quality_snapshots else 0.0
        overall_empty_rate = total_empty / total_quality_snapshots if total_quality_snapshots else 0.0
        overall_monotonic_rate = total_monotonic / total_quality_snapshots if total_quality_snapshots else 0.0
        overall_quality_score = (
            weighted_quality_score / sum(max(1, instrument.quality_snapshot_count) for instrument in instrument_results)
            if instrument_results
            else 100.0
        )

        top_gap_offenders = [
            QualityOffenderItem(
                instrument_id=instrument.instrument_id,
                instrument_type=instrument.instrument_type,
                quality_score=instrument.quality_score,
                max_gap_seconds=instrument.max_gap_seconds,
                crossed_rate=instrument.crossed_rate,
                empty_rate=instrument.empty_rate,
                bad_rate=(instrument.quality_bad_snapshot_count / instrument.quality_snapshot_count) if instrument.quality_snapshot_count else 0.0,
                snapshot_count=instrument.quality_snapshot_count,
            )
            for instrument in sorted(
                instrument_results,
                key=lambda item: item.max_gap_seconds or 0.0,
                reverse=True,
            )[:10]
        ]
        top_crossed_offenders = [
            QualityOffenderItem(
                instrument_id=instrument.instrument_id,
                instrument_type=instrument.instrument_type,
                quality_score=instrument.quality_score,
                max_gap_seconds=instrument.max_gap_seconds,
                crossed_rate=instrument.crossed_rate,
                empty_rate=instrument.empty_rate,
                bad_rate=(instrument.quality_bad_snapshot_count / instrument.quality_snapshot_count) if instrument.quality_snapshot_count else 0.0,
                snapshot_count=instrument.quality_snapshot_count,
            )
            for instrument in sorted(
                instrument_results,
                key=lambda item: item.crossed_rate,
                reverse=True,
            )[:10]
        ]
        top_empty_offenders = [
            QualityOffenderItem(
                instrument_id=instrument.instrument_id,
                instrument_type=instrument.instrument_type,
                quality_score=instrument.quality_score,
                max_gap_seconds=instrument.max_gap_seconds,
                crossed_rate=instrument.crossed_rate,
                empty_rate=instrument.empty_rate,
                bad_rate=(instrument.quality_bad_snapshot_count / instrument.quality_snapshot_count) if instrument.quality_snapshot_count else 0.0,
                snapshot_count=instrument.quality_snapshot_count,
            )
            for instrument in sorted(
                instrument_results,
                key=lambda item: item.empty_rate,
                reverse=True,
            )[:10]
        ]

        return AuditSummary(
            instrument_count=len(instrument_results),
            data_type_coverage=data_type_coverage,
            total_row_counts=total_row_counts,
            l2_sampled_snapshot_count=l2_sampled_snapshot_count,
            l2_bad_count=l2_bad_count,
            l2_bad_instrument_count=l2_bad_instrument_count,
            l2_bad_rate=l2_bad_rate,
            overall_crossed_rate=overall_crossed_rate,
            overall_empty_rate=overall_empty_rate,
            overall_monotonic_rate=overall_monotonic_rate,
            overall_quality_score=overall_quality_score,
            session_break_count=session_break_count,
            chart_points=chart_points,
            top_gap_offenders=top_gap_offenders,
            top_crossed_offenders=top_crossed_offenders,
            top_empty_offenders=top_empty_offenders,
        )

    def run_audit(
        self,
        *,
        cache_path: Path | str | None = None,
        first_n: int = 10,
        random_n: int = 10,
        progress_callback: ProgressCallback | None = None,
    ) -> AuditResponse:
        instrument_index, event_maps, warnings = self._collect_catalog_state()
        inventory = self.scan_inventory()
        instrument_ids = [instrument.instrument_id for instrument in inventory.instruments]

        total_steps = sum(
            1
            for instrument_id in instrument_ids
            for data_type in EVENT_DATA_TYPES
            if event_maps[data_type].get(instrument_id)
        )
        total_steps += sum(1 for instrument_id in instrument_ids if event_maps["order_book_depths"].get(instrument_id))
        total_steps = max(total_steps, 1)

        completed_steps = 0
        if progress_callback is not None:
            progress_callback("inventory", completed_steps, total_steps, "Inventory beolvasása kész, audit indul.")

        instrument_results: list[AuditInstrumentResult] = []

        for instrument_id in instrument_ids:
            instrument_type = instrument_index[instrument_id]
            data_type_results: dict[str, DataTypeAuditStats] = {}

            for data_type in EVENT_DATA_TYPES:
                files = event_maps[data_type].get(instrument_id, [])
                if files and progress_callback is not None:
                    progress_callback(
                        "scan",
                        completed_steps,
                        total_steps,
                        f"{instrument_id} / {data_type} összegzés fut...",
                    )

                data_type_results[data_type] = self._scan_data_type(data_type, files)
                if files:
                    completed_steps += 1
                    if progress_callback is not None:
                        progress_callback(
                            "scan",
                            completed_steps,
                            total_steps,
                            f"{instrument_id} / {data_type} összegzés kész.",
                        )

            depth_files = event_maps["order_book_depths"].get(instrument_id, [])
            if depth_files and progress_callback is not None:
                progress_callback(
                    "l2",
                    completed_steps,
                    total_steps,
                    f"{instrument_id} L2 sanity check fut...",
                )

            l2_result = (
                self._run_l2_check(
                    instrument_id,
                    first_n=first_n,
                    random_n=random_n,
                    instrument_type=instrument_type,
                    warnings=warnings,
                )
                if depth_files
                else L2CheckResult(present=False)
            )

            full_quality = None
            if depth_files:
                try:
                    full_quality = self.query_service.get_l2_quality(instrument_id)
                except Exception as exc:
                    warnings.append(
                        self._warning(
                            "l2_quality_failed",
                            f"Teljes L2 quality aggregáció nem sikerült {instrument_id}: {exc}",
                        ),
                    )

            if depth_files:
                completed_steps += 1
                if progress_callback is not None:
                    progress_callback(
                        "l2",
                        completed_steps,
                        total_steps,
                        f"{instrument_id} L2 sanity check kész.",
                    )

            instrument_results.append(
                AuditInstrumentResult(
                    instrument_id=instrument_id,
                    instrument_type=instrument_type,
                    data_types=data_type_results,
                    l2_check=l2_result,
                    quality_score=full_quality.quality_score if full_quality is not None else l2_result.quality_score,
                    quality_snapshot_count=full_quality.snapshot_count if full_quality is not None else l2_result.sampled_count,
                    quality_bad_snapshot_count=full_quality.bad_snapshot_count if full_quality is not None else l2_result.bad_count,
                    crossed_count=full_quality.crossed_count if full_quality is not None else l2_result.crossed_count,
                    monotonic_violation_count=(
                        full_quality.monotonic_violation_count if full_quality is not None else l2_result.monotonic_violation_count
                    ),
                    empty_side_count=full_quality.empty_side_count if full_quality is not None else l2_result.empty_side_count,
                    crossed_rate=full_quality.crossed_rate if full_quality is not None else 0.0,
                    empty_rate=full_quality.empty_side_rate if full_quality is not None else 0.0,
                    session_break_count=full_quality.session_break_count if full_quality is not None else 0,
                    max_gap_seconds=full_quality.max_gap_seconds if full_quality is not None else data_type_results["order_book_depths"].max_gap_seconds,
                    corrupt=any(
                        stat.corrupt for stat in data_type_results.values()
                    ) or bool(full_quality.error if full_quality is not None else False),
                ),
            )

        summary = self._build_summary(instrument_results)
        target_cache = Path(cache_path).expanduser().resolve() if cache_path else self.cache_path
        audit = AuditResponse(
            catalog_root=str(self.catalog_root),
            generated_at=_utc_now_iso(),
            cache_path=str(target_cache),
            inventory=inventory,
            instruments=instrument_results,
            summary=summary,
            warnings=_dedupe_warnings(warnings + inventory.warnings),
        )
        self.save_audit_cache(audit, target_cache)
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
