# Nautilus Catalog Viewer

FastAPI + Jinja2 + Tailwind CDN + Plotly.js CDN alapú localhost web app és CLI egy NautilusTrader `ParquetDataCatalog` böngészésére, auditálására és vizsgálható explorer/quality nézetben történő elemzésére.

## Telepítés

```bash
pip install -r requirements.txt
```

## Futtatás

Web UI:

```bash
python -m app serve --host 127.0.0.1 --port 8000 --catalog /home/zsom/services/nautilus_data/catalog
```

CLI audit:

```bash
python -m app audit --catalog /home/zsom/services/nautilus_data/catalog --out state/audit.json
```

## Oldalak

- `/`: home dashboard, inventory summary, sortable coverage tábla, audit summary, run checks
- `/instrument/{instrument_id}`: explorer nézet instrumentenként, idősáv szűrés, trade/depth chartok, snapshot viewer, quality panel
- `/quality`: audit cache alapú top offenders és per-instrument quality összefoglaló

## API endpointok

- `GET /api/inventory?search=...`
- `GET /api/instruments?type=currency_pair|crypto_perpetual&q=...`
- `GET /api/coverage?instrument_id=...&from=...&to=...`
- `GET /api/trades?instrument_id=...&from=...&to=...&mode=raw|agg&bucket_s=60&max_points=10000`
- `GET /api/l2/timeseries?instrument_id=...&from=...&to=...&mode=raw|agg&bucket_s=60&max_points=10000`
- `GET /api/l2/snapshot?instrument_id=...&ts=...`
- `GET /api/l2/snapshot?instrument_id=...&index=123`
- `GET /api/l2/quality?instrument_id=...&from=...&to=...`
- `GET /api/export?instrument_id=...&from=...&to=...&kind=trades|l2|bundle`
- `GET /api/audit`
- `POST /api/audit/run`
- `GET /api/progress`

## WSL2 és performance tippek

- A `--catalog` paraméter legyen Linux path, például: `/home/zsom/services/nautilus_data/catalog`
- A webes audit cache alapértelmezett helye: `./state/web_audit_cache.json`
- Az API disk cache könyvtára: `./state/api_cache/`
- Nagyobb idősávnál használd az aggregált módot:

```text
/api/trades?...&mode=agg&bucket_s=60
/api/l2/timeseries?...&mode=agg&bucket_s=60
```

- Ha a chart túl sűrű, csökkentsd a pontszámot:

```text
...&max_points=2000
```

- A backend `pyarrow.dataset` filter pushdownt használ `ts_event` alapján, ezért a `from`/`to` szűrés érdemben gyorsít

## Debug bundle export

- Az explorer oldalon az `Export debug bundle` gomb a kiválasztott idősávra JSON bundle-t tölt le
- A bundle tartalmazza:
  - coverage summary
  - trade raw pontok
  - L2 quality summary
  - L2 snapshotok top10 bids/asks + size tömbökkel
- Hibás snapshot vizsgálatára állíts szűk `from`/`to` ablakot, majd exportáld a bundle-t

## Megjegyzések

- Instrument listázáshoz és a meglévő audit kompatibilitáshoz a rendszer használja a `nautilus_trader.persistence.catalog.ParquetDataCatalog`-ot, ahol érdemes
- A row count, time-range, aggregation és snapshot queryk `pyarrow.dataset` alapon mennek
- Ha egy adott instrument parquet állománya hibás, az endpoint vagy audit rekord `error/corrupt` mezővel jelöli, és a szerver a többi instrumenttel továbbmegy
