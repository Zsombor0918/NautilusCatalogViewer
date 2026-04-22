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
pip install -r requirements.txt
```

Optional: `nautilus_trader>=1.225` is used for `ParquetDataCatalog`-based
instrument listing when available. All core queries use `pyarrow.dataset`.

## Running

Web UI:

```bash
python -m app serve --host 127.0.0.1 --port 8000 --catalog /path/to/catalog
```

CLI audit:

```bash
python -m app audit --catalog /path/to/catalog --out state/audit.json
```

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
| `GET` | `/api/readiness?instrument_id=...` | Readiness assessment: score, fenced ranges, desyncs, backtest-ready flag |

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

## Readiness Model

Each instrument receives a **readiness score** (0–100) based on:

| Component | Weight | Condition |
|---|---|---|
| Trade data present | +20 | `trade_tick` directory exists with data |
| Delta data present | +25 | `order_book_deltas` directory exists |
| Depth10 data present | +5 | `order_book_depths` (optional bonus) |
| Trade row volume | +10 | ≥1000 rows |
| Delta row volume | +10 | ≥1000 rows |
| No trade gaps | +10 | max gap < 60s |
| No delta gaps | +10 | max gap < 60s |
| No fenced ranges | +5 | 0 fenced ranges from report |
| No desyncs | +5 | 0 desync events from report |

**Penalties** are subtracted for fenced ranges (−3 each), desyncs (−5 each),
excessive resyncs (−1 per 5), and session breaks (−1 per 3).

An instrument is **backtest-ready** when:
- `readiness_score ≥ 60`
- `has_trade_tick` and `has_order_book_deltas` are both true
- `desync_count == 0`

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
pytest tests/ -v
```

## Architecture Notes

- Uses `nautilus_trader.persistence.catalog.ParquetDataCatalog` for instrument listing when available
- All row count, time-range, aggregation, and snapshot queries use `pyarrow.dataset` directly
- Corrupt parquet files are marked as `error/corrupt` in audit records; the server continues with other instruments
- L2 checks (`l2_checks.py`) are retained for optional depth10 quality analysis but are not part of the primary readiness model
