from __future__ import annotations

import argparse
from pathlib import Path

import uvicorn

from .catalog_scan import CatalogScanner
from .main import DEFAULT_CACHE_PATH, DEFAULT_CATALOG_ROOT, DEFAULT_CONVERTER_REPORTS_DIR, create_app


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m app",
        description="Nautilus ParquetDataCatalog localhost viewer and auditor.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    serve_parser = subparsers.add_parser("serve", help="Start FastAPI web app.")
    serve_parser.add_argument("--host", default="127.0.0.1")
    serve_parser.add_argument("--port", type=int, default=8000)
    serve_parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG_ROOT)
    serve_parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE_PATH)
    serve_parser.add_argument("--converter-reports", type=Path, default=DEFAULT_CONVERTER_REPORTS_DIR, dest="converter_reports",
                              help="Path to converter reports directory (contains YYYY-MM-DD.json files).")

    audit_parser = subparsers.add_parser("audit", help="Run CLI audit.")
    audit_parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG_ROOT)
    audit_parser.add_argument("--out", type=Path, default=Path("state/audit.json"))
    audit_parser.add_argument("--converter-reports", type=Path, default=DEFAULT_CONVERTER_REPORTS_DIR, dest="converter_reports",
                              help="Path to converter reports directory (contains YYYY-MM-DD.json files).")
    audit_parser.add_argument(
        "--first-n", type=int, default=10,
        help="Sample first N depth snapshots.",
    )
    audit_parser.add_argument(
        "--random-n", type=int, default=10,
        help="Sample random N depth snapshots.",
    )

    return parser


def serve_command(args: argparse.Namespace) -> int:
    app = create_app(catalog_root=args.catalog, cache_path=args.cache, converter_reports_dir=args.converter_reports)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0


def audit_command(args: argparse.Namespace) -> int:
    scanner = CatalogScanner(catalog_root=args.catalog, cache_path=args.out, converter_reports_dir=args.converter_reports)
    audit = scanner.run_audit(
        cache_path=args.out,
        first_n=args.first_n,
        random_n=args.random_n,
    )

    s = audit.summary
    dtc = s.data_type_coverage

    print(f"Audit complete: {audit.catalog_root}")
    print(f"Instruments: {s.instrument_count}")
    print(f"trade_tick coverage: {dtc.get('trade_tick', 0)}")
    print(f"order_book_deltas coverage: {dtc.get('order_book_deltas', 0)}")
    print(f"order_book_depths coverage (optional): {dtc.get('order_book_depths', 0)}")
    print(f"backtest_ready_count: {s.backtest_ready_count}")
    print(f"consumable_count: {s.consumable_count}")
    print(f"avg_readiness_score: {s.avg_readiness_score:.2f}")
    print(f"total_fenced_range_count: {s.total_fenced_range_count}")
    print(f"total_desync_count: {s.total_desync_count}")
    print(f"Output: {Path(args.out).expanduser().resolve()}")

    return 0


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "serve":
        return serve_command(args)
    if args.command == "audit":
        return audit_command(args)

    parser.error("Unknown command.")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
