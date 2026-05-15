# Nautilus Catalog Viewer

**v0.3.0 — Deterministic-first catalog explorer and audit tool**

A FastAPI + Jinja2 + Tailwind CDN + Plotly.js web app and CLI for browsing,
auditing, and assessing readiness of a NautilusTrader `ParquetDataCatalog`.

The primary data model is **TradeTick + OrderBookDeltas** — the deterministic
replay pair required by `BacktestNode`. OrderBookDepth10 (L2 snapshots) is
supported as an optional secondary data type for visual quality inspection.

Think of it as a **Tardis-like** inspector: the recorder/converter writes
deterministic deltas with fenced ranges and resync markers, and this viewer
lets you verify that the catalog is replay-ready before feeding it to Nautilus.

## Installation

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

Optional: `nautilus_trader>=1.225` is used for `ParquetDataCatalog`-based
instrument listing when available. All core queries use `pyarrow.dataset`.

## Running

The repo includes a local config at `config/viewer.env` pointing at:

```bash
NAUTILUS_CATALOG_ROOT=/home/zsom/sync/catalog
NAUTILUS_VIEWER_CONVERT_REPORT_DIR=/home/zsom/sync/convert_reports
```

Web UI with the configured catalog:

```bash
.venv/bin/python main.py
```

The package entrypoint uses the same defaults:

```bash
.venv/bin/python -m app serve
```

CLI audit:

```bash
.venv/bin/python -m app audit --out state/audit.json
```

Parquet metadata debug:

```bash
.venv/bin/python -m app debug-parquet --catalog /path/to/catalog --instrument BTCUSDT-PERP.BINANCE --data-type order_book_deltas
```

Optional convert report selection:

```bash
.venv/bin/python -m app audit --catalog /path/to/catalog --convert-report-dir /path/to/convert_reports --convert-report-date 2026-04-25
```

To use a different config file, set `NAUTILUS_VIEWER_CONFIG=/path/to/viewer.env`
before launching the app.

## Pages

| Route | Description |
|---|---|
| `/` | Dashboard — inventory summary, delta coverage table, readiness stats, audit controls |
| `/instrument/{id}` | Explorer — time-range filtering, trade/delta/depth charts, readiness panel, report context |
| `/readiness` | Readiness overview — per-instrument readiness scores, backtest-ready status, offenders |

> The former `/quality` route redirects to `/readiness` (HTTP 301).

## API Endpoints

### Core

| Method | Path | Description |
|---|---|---|
| `GET` | `/api/inventory?search=...` | Inventory listing with delta/trade/depth coverage |
| `GET` | `/api/instruments?type=...&q=...` | Instrument search with `has_order_book_deltas` flag |
| `GET` | `/api/coverage?instrument_id=...&from=...&to=...` | Per-type coverage with row counts and time ranges |
| `GET` | `/api/trades?instrument_id=...&mode=raw\|agg&bucket_s=60` | Trade data (raw or aggregated) |

### Deltas (new)

| Method | Path | Description |
|---|---|---|
| `GET` | `/api/deltas/summary?instrument_id=...` | Delta file stats: row count, time range, action distribution |
| `GET` | `/api/deltas?instrument_id=...&mode=raw\|agg&bucket_s=60` | Delta events (raw or aggregated into buckets) |

### Readiness (new, replaces quality)

| Method | Path | Description |
|---|---|---|
| `GET` | `/api/readiness?instrument_id=...` | Backtest readiness assessment: data completeness for Nautilus |

### L2 (optional, depth10 only)

| Method | Path | Description |
|---|---|---|
| `GET` | `/api/l2/timeseries?instrument_id=...&mode=raw\|agg` | L2 depth10 time series |
| `GET` | `/api/l2/snapshot?instrument_id=...&ts=...` | Single L2 snapshot at timestamp |
| `GET` | `/api/l2/quality?instrument_id=...` | L2 quality checks (crossed/empty/gap analysis) |

### Export & Audit

| Method | Path | Description |
|---|---|---|
| `GET` | `/api/export?instrument_id=...&kind=trades\|l2\|deltas\|bundle` | Data export (JSON). Bundle includes coverage + trades + deltas_summary + readiness |
| `GET` | `/api/audit` | Latest audit results |
| `POST` | `/api/audit/run` | Trigger background audit |
| `GET` | `/api/progress` | Audit progress polling |

## Score Model

The audit exposes three separate scores:

| Score | Meaning |
|---|---|
| `backtest_readiness_score` | Data completeness for Nautilus backtests |
| `l2_quality_score` / `data_reliability_score` | Crossed books, monotonic violations, quantity issues, gaps, session breaks, fenced ranges, desync/resync signals, bad lines, and missing/partial symbols |
| `audit_confidence_score` | How much to trust the scan: unreadable row counts, null timestamp bounds, missing/stale convert report, and audit/convert mismatch |

Backtest readiness is intentionally simple:

| Status | Score | Condition |
|---|---:|---|
| `full_ready` | 100 | `TradeTick` and `OrderBookDeltas` present |
| `l2_ready` | 70 | L2 data present without full TradeTick + OrderBookDeltas completeness |
| `trade_ready` | 60 | `TradeTick` present without L2 data |
| `partial_unreadable` | 40 | Files exist but row counts/timestamps are unreadable |
| `not_ready` | 0 | No usable replay data |

An instrument is **backtest-ready** when `TradeTick` and `OrderBookDeltas`
are both present. Fences, desyncs, gaps and session breaks remain visible under
reliability issues and lower `l2_quality_score`, not the completeness score.

Each audited data type also has an explicit `status`:

| Status | Meaning |
|---|---|
| `absent` | No parquet files found |
| `present_empty` | Files exist and parquet metadata confirms zero rows |
| `present_with_rows` | Files exist and parquet metadata reports rows |
| `present_unreadable` | Files exist but metadata/schema/row scan failed |
| `present_unknown_rows` | Files exist but row count cannot be trusted |

Timestamp issues are tracked separately with `timestamp_status`, so missing
timestamp columns no longer erase metadata row counts.

## Deterministic Report Integration

If the catalog contains a `reports/` directory with per-instrument JSON files
(e.g. `reports/BTCUSDT-PERP.BINANCE.json`), the viewer ingests:

- **Fenced ranges** — time windows excluded from replay (exchange maintenance, etc.)
- **Session boundaries** — recording session start/end markers
- **Resync events** — snapshot seeds, resyncs, desyncs with timestamps
- **Converter diagnostics** — trade ID duplicates, slow throughput warnings
- **Last committed update ID** — the final acknowledged exchange update

These reports are produced by the deterministic recorder/converter and are
displayed in the instrument explorer's "Report Context" section.

## Convert Report Discovery

CryptoRecorder convert reports are discovered in this order:

1. Explicit CLI/API argument: `--convert-report-dir`
2. Environment variable: `NAUTILUS_VIEWER_CONVERT_REPORT_DIR`
3. Sibling folder of the catalog root: `catalog_root.parent / "convert_reports"`
4. App-local fallback: `./state/convert_reports`

If no date is specified, the latest `YYYY-MM-DD.json` filename is selected.
Use `--convert-report-date YYYY-MM-DD` to load a specific report. The audit
summary exposes the selected report path, date, timestamp, status,
`catalog_root` match, `report_paths`, and warnings.

## Debug Bundle Export

The instrument explorer's "Export debug bundle" button downloads a JSON bundle
containing:

- Coverage summary (all three data types)
- Raw trade points for the selected time range
- Deltas summary (action/side distribution, row counts)
- Readiness assessment (score, fenced ranges, desyncs, backtest-ready flag)
- L2 quality summary (only if depth10 is present)
- L2 snapshots with top-10 bids/asks (only if depth10 is present)

## Performance Notes

- Use Linux paths for `--catalog` (important on WSL2)
- Web audit cache: `./state/web_audit_cache.json`
- API disk cache: `./state/api_cache/`
- Use aggregated mode for large time ranges: `&mode=agg&bucket_s=60`
- Reduce chart density: `&max_points=2000`
- Backend uses `pyarrow.dataset` filter pushdown on `ts_event` — `from`/`to` filtering is efficient

## Testing

```bash
.venv/bin/python -m pip install -r requirements-dev.txt
.venv/bin/python -m pytest tests/ -v
```

## Architecture Notes

- Uses `nautilus_trader.persistence.catalog.ParquetDataCatalog` for instrument listing when available
- All row count, time-range, aggregation, and snapshot queries use `pyarrow.dataset` directly
- Corrupt parquet files are marked as `error/corrupt` in audit records; the server continues with other instruments
- L2 checks (`l2_checks.py`) are retained for optional depth10 quality analysis but are not part of the primary readiness model
