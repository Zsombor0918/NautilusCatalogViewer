from __future__ import annotations

import argparse
from pathlib import Path

import uvicorn

from .catalog_scan import CatalogScanner
from .main import DEFAULT_CACHE_PATH, DEFAULT_CATALOG_ROOT, create_app


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m app",
        description="Nautilus ParquetDataCatalog localhost viewer and auditor.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    serve_parser = subparsers.add_parser("serve", help="FastAPI web app indítása.")
    serve_parser.add_argument("--host", default="127.0.0.1")
    serve_parser.add_argument("--port", type=int, default=8000)
    serve_parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG_ROOT)
    serve_parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE_PATH)

    audit_parser = subparsers.add_parser("audit", help="CLI audit futtatása.")
    audit_parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG_ROOT)
    audit_parser.add_argument("--out", type=Path, default=Path("state/audit.json"))
    audit_parser.add_argument("--first-n", type=int, default=10, help="Első N depth snapshot mintavétele.")
    audit_parser.add_argument("--random-n", type=int, default=10, help="Random N depth snapshot mintavétele.")

    return parser


def serve_command(args: argparse.Namespace) -> int:
    app = create_app(catalog_root=args.catalog, cache_path=args.cache)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0


def audit_command(args: argparse.Namespace) -> int:
    scanner = CatalogScanner(catalog_root=args.catalog, cache_path=args.out)
    audit = scanner.run_audit(
        cache_path=args.out,
        first_n=args.first_n,
        random_n=args.random_n,
    )
    print(f"Audit kész: {audit.catalog_root}")
    print(f"Instrumentek: {audit.summary.instrument_count}")
    print(f"trade_tick coverage: {audit.summary.data_type_coverage['trade_tick']}")
    print(f"order_book_depths coverage: {audit.summary.data_type_coverage['order_book_depths']}")
    print(f"L2 sampled snapshots: {audit.summary.l2_sampled_snapshot_count}")
    print(f"L2 bad snapshots: {audit.summary.l2_bad_count}")
    print(f"Overall crossed rate: {audit.summary.overall_crossed_rate:.4%}")
    print(f"Overall quality score: {audit.summary.overall_quality_score:.2f}")
    print(f"Kimenet: {Path(args.out).expanduser().resolve()}")
    return 0


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "serve":
        return serve_command(args)
    if args.command == "audit":
        return audit_command(args)

    parser.error("Ismeretlen parancs.")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
