from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, RedirectResponse, StreamingResponse
from fastapi.templating import Jinja2Templates

from .catalog_scan import CatalogScanner
from .models import (
    AuditResponse,
    CoverageResponse,
    DeltasResponse,
    DeltasSummaryResponse,
    Depth10DebugResponse,
    InstrumentSearchResponse,
    InventoryResponse,
    L2QualityResponse,
    L2SnapshotResponse,
    L2TimeseriesResponse,
    ProgressResponse,
    ReadinessResponse,
    ReadinessResult,
    TradesResponse,
)
from .query import CatalogQueryService


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CATALOG_ROOT = Path(
    os.getenv("NAUTILUS_CATALOG_ROOT", "/home/zsomb/nautilus_data/catalog"),
).expanduser()
DEFAULT_CACHE_PATH = PROJECT_ROOT / "state" / "web_audit_cache.json"
_converter_env = os.getenv("NAUTILUS_VIEWER_CONVERT_REPORT_DIR") or os.getenv("NAUTILUS_CONVERTER_REPORTS_DIR")
DEFAULT_CONVERTER_REPORTS_DIR: Path | None = Path(_converter_env).expanduser() if _converter_env else None
DEFAULT_CONVERT_REPORT_DATE: str | None = os.getenv("NAUTILUS_VIEWER_CONVERT_REPORT_DATE")
TEMPLATES = Jinja2Templates(directory=str(PROJECT_ROOT / "templates"))


# ── Jinja2 template filters ────────────────────────────────────────────────────────────────────────


def _format_int(value: Any) -> str:
    if value is None:
        return "—"
    return format(int(value), ",").replace(",", " ")


def _format_seconds(value: Any) -> str:
    if value is None:
        return "—"
    return f"{float(value):,.3f} s".replace(",", " ")


def _format_pct(value: Any) -> str:
    if value is None:
        return "—"
    return f"{float(value) * 100:.2f}%"


def _format_progress_pct(value: Any) -> str:
    if value is None:
        return "0%"
    return f"{float(value):.0f}%"


def _format_float(value: Any) -> str:
    if value is None:
        return "—"
    return f"{float(value):,.6f}".replace(",", " ")


def _utc_now_iso() -> str:
    return datetime.now(tz=timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


TEMPLATES.env.filters["format_int"] = _format_int
TEMPLATES.env.filters["format_seconds"] = _format_seconds
TEMPLATES.env.filters["format_pct"] = _format_pct
TEMPLATES.env.filters["format_progress_pct"] = _format_progress_pct
TEMPLATES.env.filters["format_float"] = _format_float


# ── Runtime state ────────────────────────────────────────────────────────────────────────────


@dataclass
class RuntimeState:
    lock: threading.Lock = field(default_factory=threading.Lock)
    progress: ProgressResponse = field(default_factory=ProgressResponse)
    latest_audit: AuditResponse | None = None
    worker: threading.Thread | None = None


# ── Helpers ─────────────────────────────────────────────────────────────────────────────────


def _percent(completed_steps: int, total_steps: int) -> float:
    if total_steps <= 0:
        return 0.0
    return min(100.0, round((completed_steps / total_steps) * 100, 2))


def _update_progress(
    runtime: RuntimeState,
    *,
    status: Literal["idle", "running", "completed", "failed"],
    phase: str,
    message: str,
    completed_steps: int,
    total_steps: int,
    started_at: str | None,
    finished_at: str | None = None,
    cache_path: str | None = None,
    error: str | None = None,
) -> ProgressResponse:
    progress = ProgressResponse(
        status=status,
        phase=phase,
        message=message,
        completed_steps=completed_steps,
        total_steps=total_steps,
        percent=_percent(completed_steps, total_steps),
        started_at=started_at,
        finished_at=finished_at,
        cache_path=cache_path,
        error=error,
    )
    runtime.progress = progress
    return progress


def _load_audit(app: FastAPI) -> AuditResponse | None:
    runtime: RuntimeState = app.state.runtime
    scanner: CatalogScanner = app.state.scanner
    if runtime.latest_audit is None:
        runtime.latest_audit = scanner.load_audit_cache()
    return runtime.latest_audit


def _default_instrument_range(audit: AuditResponse | None, instrument_id: str) -> tuple[str | None, str | None]:
    if audit is None:
        return None, None
    match = next((item for item in audit.instruments if item.instrument_id == instrument_id), None)
    if match is None:
        return None, None

    start_candidates = [
        stats.ts_event_min_iso
        for stats in match.data_types.values()
        if stats.ts_event_min_iso is not None
    ]
    end_candidates = [
        stats.ts_event_max_iso
        for stats in match.data_types.values()
        if stats.ts_event_max_iso is not None
    ]
    return (min(start_candidates) if start_candidates else None, max(end_candidates) if end_candidates else None)


# ── Background audit worker ──────────────────────────────────────────────────────────────────


def _run_audit_worker(app: FastAPI, started_at: str) -> None:
    scanner: CatalogScanner = app.state.scanner
    runtime: RuntimeState = app.state.runtime

    def progress_callback(phase: str, completed_steps: int, total_steps: int, message: str) -> None:
        with runtime.lock:
            _update_progress(
                runtime,
                status="running",
                phase=phase,
                message=message,
                completed_steps=completed_steps,
                total_steps=total_steps,
                started_at=started_at,
                cache_path=str(scanner.cache_path),
            )

    try:
        audit = scanner.run_audit(cache_path=scanner.cache_path, progress_callback=progress_callback)
        with runtime.lock:
            runtime.latest_audit = audit
            _update_progress(
                runtime,
                status="completed",
                phase="done",
                message="Audit complete, cache file updated.",
                completed_steps=max(1, runtime.progress.total_steps),
                total_steps=max(1, runtime.progress.total_steps),
                started_at=started_at,
                finished_at=audit.generated_at,
                cache_path=str(scanner.cache_path),
            )
            runtime.worker = None
    except Exception as exc:
        with runtime.lock:
            _update_progress(
                runtime,
                status="failed",
                phase="failed",
                message="Audit stopped due to an error.",
                completed_steps=runtime.progress.completed_steps,
                total_steps=runtime.progress.total_steps,
                started_at=started_at,
                finished_at=_utc_now_iso(),
                cache_path=str(scanner.cache_path),
                error=str(exc),
            )
            runtime.worker = None


# ── Application factory ──────────────────────────────────────────────────────────────────────


def create_app(
    catalog_root: Path | str = DEFAULT_CATALOG_ROOT,
    cache_path: Path | str = DEFAULT_CACHE_PATH,
    converter_reports_dir: Path | str | None = DEFAULT_CONVERTER_REPORTS_DIR,
    convert_report_date: str | None = DEFAULT_CONVERT_REPORT_DATE,
) -> FastAPI:
    app = FastAPI(title="Nautilus Catalog Viewer", version="0.3.0", docs_url="/docs", redoc_url="/redoc")
    scanner = CatalogScanner(catalog_root=catalog_root, cache_path=cache_path, converter_reports_dir=converter_reports_dir, convert_report_date=convert_report_date)
    query_service: CatalogQueryService = scanner.query_service
    runtime = RuntimeState()
    runtime.latest_audit = scanner.load_audit_cache()
    if runtime.latest_audit is not None:
        runtime.progress = ProgressResponse(
            status="completed",
            phase="idle",
            message="Cached audit loaded.",
            completed_steps=1,
            total_steps=1,
            percent=100.0,
            finished_at=runtime.latest_audit.generated_at,
            cache_path=str(scanner.cache_path),
        )

    app.state.scanner = scanner
    app.state.query_service = query_service
    app.state.runtime = runtime

    # ── HTML pages ──────────────────────────────────────────────────────────────────────

    @app.get("/", response_class=HTMLResponse)
    async def index(request: Request) -> HTMLResponse:
        inventory = app.state.scanner.scan_inventory()
        audit = _load_audit(app)
        chart_payload = json.dumps(
            [point.model_dump(mode="json") for point in (audit.summary.chart_points if audit else [])],
            ensure_ascii=False,
        )
        inventory_payload = json.dumps(
            [item.model_dump(mode="json") for item in inventory.instruments],
            ensure_ascii=False,
        )
        # Readiness summary from audit
        backtest_ready_count = audit.summary.backtest_ready_count if audit else 0
        consumable_count = audit.summary.consumable_count if audit else 0
        avg_readiness_score = audit.summary.avg_readiness_score if audit else 0.0
        return TEMPLATES.TemplateResponse(
            request=request,
            name="index.html",
            context={
                "inventory": inventory,
                "audit": audit,
                "progress": app.state.runtime.progress,
                "cache_path": str(app.state.scanner.cache_path),
                "chart_payload": chart_payload,
                "inventory_payload": inventory_payload,
                "backtest_ready_count": backtest_ready_count,
                "consumable_count": consumable_count,
                "avg_readiness_score": avg_readiness_score,
            },
        )

    @app.get("/instrument/{instrument_id}", response_class=HTMLResponse)
    async def instrument_explorer(request: Request, instrument_id: str) -> HTMLResponse:
        inventory = app.state.scanner.scan_inventory()
        instrument = next((item for item in inventory.instruments if item.instrument_id == instrument_id), None)
        if instrument is None:
            raise HTTPException(status_code=404, detail=f"Instrument not found: {instrument_id}")

        audit = _load_audit(app)
        default_from_iso, default_to_iso = _default_instrument_range(audit, instrument_id)
        inventory_payload = json.dumps(
            [item.model_dump(mode="json") for item in inventory.instruments],
            ensure_ascii=False,
        )

        # Readiness info for this instrument
        readiness: ReadinessResult = ReadinessResult(
            instrument_id=instrument_id,
            instrument_type=instrument.instrument_type,
        )
        deltas_summary = DeltasSummaryResponse(
            instrument_id=instrument_id,
            instrument_type=instrument.instrument_type,
            generated_at=_utc_now_iso(),
        )
        try:
            readiness = query_service.get_readiness(instrument_id).readiness
        except Exception:
            pass
        audit_item = next((item for item in audit.instruments if item.instrument_id == instrument_id), None) if audit else None
        try:
            deltas_summary = query_service.get_deltas_summary(instrument_id)
        except Exception:
            pass

        # Build a merged display_readiness: live readiness for scores/row counts,
        # audit_item.readiness for converter-report fields (fenced ranges, resync, etc.).
        # This eliminates the contradiction where live scores show 0 fenced ranges
        # while audit suggestions show e.g. 3 fenced ranges from the converter report.
        display_readiness = readiness
        readiness_source = "catalog scan"
        if audit_item and audit_item.readiness.instrument_id:
            ar = audit_item.readiness
            merged = readiness.model_dump()
            merged["fenced_range_count"] = ar.fenced_range_count if ar.converter_report_found else readiness.fenced_range_count
            merged["fenced_ranges_by_reason"] = ar.fenced_ranges_by_reason or readiness.fenced_ranges_by_reason
            merged["resync_count"] = ar.resync_count if ar.converter_report_found else readiness.resync_count
            merged["desync_count"] = ar.desync_count if ar.converter_report_found else readiness.desync_count
            merged["snapshot_seed_count"] = ar.snapshot_seed_count if ar.converter_report_found else readiness.snapshot_seed_count
            merged["converter_report_found"] = ar.converter_report_found or readiness.converter_report_found
            merged["report"] = (ar.report if ar.report.report_found else readiness.report).model_dump()
            display_readiness = ReadinessResult(**merged)
            readiness_source = "converter report" if ar.converter_report_found else "catalog scan"

        return TEMPLATES.TemplateResponse(
            request=request,
            name="instrument.html",
            context={
                "inventory": inventory,
                "instrument": instrument,
                "audit": audit,
                "default_from_iso": default_from_iso,
                "default_to_iso": default_to_iso,
                "inventory_payload": inventory_payload,
                "readiness": display_readiness,
                "readiness_source": readiness_source,
                "audit_item": audit_item,
                "deltas_summary": deltas_summary,
            },
        )

    @app.get("/quality", response_class=RedirectResponse)
    async def quality_redirect() -> RedirectResponse:
        return RedirectResponse(url="/readiness", status_code=301)

    @app.get("/readiness", response_class=HTMLResponse)
    async def readiness_page(request: Request) -> HTMLResponse:
        inventory = app.state.scanner.scan_inventory()
        audit = _load_audit(app)
        sorted_instruments = sorted(
            audit.instruments if audit else [],
            key=lambda item: (item.readiness.readiness_score, -(item.readiness.delta_row_count or 0)),
        )
        return TEMPLATES.TemplateResponse(
            request=request,
            name="quality.html",
            context={
                "inventory": inventory,
                "audit": audit,
                "sorted_instruments": sorted_instruments,
                "progress": app.state.runtime.progress,
            },
        )

    # ── JSON API endpoints ──────────────────────────────────────────────────────────────────

    @app.get("/api/inventory", response_model=InventoryResponse)
    async def inventory_api(search: str | None = None) -> InventoryResponse:
        return app.state.scanner.scan_inventory(search=search)

    @app.get("/api/instruments", response_model=InstrumentSearchResponse)
    async def instruments_api(
        type_filter: str | None = Query(None, alias="type"),
        q: str | None = None,
    ) -> InstrumentSearchResponse:
        inventory = app.state.scanner.scan_inventory()
        payload = [item.model_dump(mode="json") for item in inventory.instruments]
        return app.state.query_service.get_instruments(payload, type_filter=type_filter, q=q)

    @app.get("/api/coverage", response_model=CoverageResponse)
    async def coverage_api(
        instrument_id: str,
        from_value: str | None = Query(None, alias="from"),
        to_value: str | None = Query(None, alias="to"),
    ) -> CoverageResponse:
        try:
            return app.state.query_service.get_coverage(instrument_id, from_value=from_value, to_value=to_value)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/trades", response_model=TradesResponse)
    async def trades_api(
        instrument_id: str,
        from_value: str | None = Query(None, alias="from"),
        to_value: str | None = Query(None, alias="to"),
        mode: Literal["raw", "agg"] = "raw",
        bucket_s: int = 60,
        max_points: int = 10_000,
    ) -> TradesResponse:
        try:
            return app.state.query_service.get_trades(
                instrument_id,
                from_value=from_value,
                to_value=to_value,
                mode=mode,
                bucket_s=bucket_s,
                max_points=max_points,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/deltas", response_model=DeltasResponse)
    async def deltas_api(
        instrument_id: str,
        from_value: str | None = Query(None, alias="from"),
        to_value: str | None = Query(None, alias="to"),
        mode: Literal["raw", "agg"] = "raw",
        bucket_s: int = 60,
        max_points: int = 10_000,
    ) -> DeltasResponse:
        try:
            return app.state.query_service.get_deltas(
                instrument_id,
                from_value=from_value,
                to_value=to_value,
                mode=mode,
                bucket_s=bucket_s,
                max_points=max_points,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/deltas/summary", response_model=DeltasSummaryResponse)
    async def deltas_summary_api(instrument_id: str) -> DeltasSummaryResponse:
        try:
            return app.state.query_service.get_deltas_summary(instrument_id)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/readiness", response_model=ReadinessResponse)
    async def readiness_api(instrument_id: str) -> ReadinessResponse:
        try:
            return app.state.query_service.get_readiness(instrument_id)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/l2/timeseries", response_model=L2TimeseriesResponse)
    async def l2_timeseries_api(
        instrument_id: str,
        from_value: str | None = Query(None, alias="from"),
        to_value: str | None = Query(None, alias="to"),
        mode: Literal["raw", "agg"] = "raw",
        bucket_s: int = 60,
        max_points: int = 10_000,
    ) -> L2TimeseriesResponse:
        try:
            return app.state.query_service.get_l2_timeseries(
                instrument_id,
                from_value=from_value,
                to_value=to_value,
                mode=mode,
                bucket_s=bucket_s,
                max_points=max_points,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/l2/snapshot", response_model=L2SnapshotResponse)
    async def l2_snapshot_api(
        instrument_id: str,
        ts_value: str | None = Query(None, alias="ts"),
        index: int | None = None,
        context_before: int = 5,
        context_after: int = 5,
    ) -> L2SnapshotResponse:
        try:
            return app.state.query_service.get_l2_snapshot(
                instrument_id,
                ts_value=ts_value,
                index=index,
                context_before=context_before,
                context_after=context_after,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/l2/quality", response_model=L2QualityResponse)
    async def l2_quality_api(
        instrument_id: str,
        from_value: str | None = Query(None, alias="from"),
        to_value: str | None = Query(None, alias="to"),
    ) -> L2QualityResponse:
        try:
            return app.state.query_service.get_l2_quality(
                instrument_id,
                from_value=from_value,
                to_value=to_value,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/debug/depth10", response_model=Depth10DebugResponse)
    async def debug_depth10_api(instrument_id: str) -> Depth10DebugResponse:
        return app.state.query_service.debug_depth10(instrument_id)

    @app.get("/api/export")
    async def export_api(
        instrument_id: str,
        kind: Literal["trades", "l2", "deltas", "bundle"],
        from_value: str | None = Query(None, alias="from"),
        to_value: str | None = Query(None, alias="to"),
    ) -> StreamingResponse:
        try:
            if kind == "trades":
                content = app.state.query_service.export_trades_csv(
                    instrument_id,
                    from_value=from_value,
                    to_value=to_value,
                )
                filename = f"{instrument_id}_trades.csv"
                media_type = "text/csv; charset=utf-8"
            elif kind == "l2":
                content = app.state.query_service.export_l2_csv(
                    instrument_id,
                    from_value=from_value,
                    to_value=to_value,
                )
                filename = f"{instrument_id}_l2.csv"
                media_type = "text/csv; charset=utf-8"
            elif kind == "deltas":
                deltas_resp = app.state.query_service.get_deltas(
                    instrument_id,
                    from_value=from_value,
                    to_value=to_value,
                    mode="raw",
                    max_points=50_000,
                )
                content = json.dumps(deltas_resp.model_dump(mode="json"), ensure_ascii=False, indent=2).encode("utf-8")
                filename = f"{instrument_id}_deltas.json"
                media_type = "application/json; charset=utf-8"
            else:
                # bundle
                content = app.state.query_service.export_bundle_json(
                    instrument_id,
                    from_value=from_value,
                    to_value=to_value,
                )
                filename = f"{instrument_id}_debug_bundle.json"
                media_type = "application/json; charset=utf-8"

            return StreamingResponse(
                iter([content]),
                media_type=media_type,
                headers={"Content-Disposition": f'attachment; filename="{filename}"'},
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/audit", response_model=AuditResponse)
    async def audit_api() -> AuditResponse:
        audit = _load_audit(app)
        if audit is None:
            raise HTTPException(status_code=404, detail="No audit cache available yet.")
        return audit

    @app.get("/api/progress", response_model=ProgressResponse)
    async def progress_api() -> ProgressResponse:
        return app.state.runtime.progress

    @app.post("/api/audit/run", response_model=ProgressResponse, status_code=202)
    async def run_audit_api() -> ProgressResponse:
        current_runtime: RuntimeState = app.state.runtime
        current_scanner: CatalogScanner = app.state.scanner
        with current_runtime.lock:
            if current_runtime.worker is not None and current_runtime.worker.is_alive():
                return current_runtime.progress

            started_at = _utc_now_iso()
            progress = _update_progress(
                current_runtime,
                status="running",
                phase="queued",
                message="Starting audit...",
                completed_steps=0,
                total_steps=0,
                started_at=started_at,
                cache_path=str(current_scanner.cache_path),
            )
            worker = threading.Thread(
                target=_run_audit_worker,
                args=(app, progress.started_at or started_at),
                daemon=True,
                name="catalog-audit-worker",
            )
            current_runtime.worker = worker
            worker.start()
            return progress

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    return app


app = create_app()
