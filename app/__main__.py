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
    serve_parser.add_argument("--convert-report-dir", "--converter-reports", type=Path, default=DEFAULT_CONVERTER_REPORTS_DIR, dest="converter_reports",
                              help="Path to convert reports directory (contains YYYY-MM-DD.json files).")
    serve_parser.add_argument("--convert-report-date", default=None, help="Specific convert report date to load, e.g. 2026-04-25.")

    audit_parser = subparsers.add_parser("audit", help="Run CLI audit.")
    audit_parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG_ROOT)
    audit_parser.add_argument("--out", type=Path, default=Path("state/audit.json"))
    audit_parser.add_argument("--convert-report-dir", "--converter-reports", type=Path, default=DEFAULT_CONVERTER_REPORTS_DIR, dest="converter_reports",
                              help="Path to convert reports directory (contains YYYY-MM-DD.json files).")
    audit_parser.add_argument("--convert-report-date", default=None, help="Specific convert report date to load, e.g. 2026-04-25.")
    audit_parser.add_argument(
        "--first-n", type=int, default=10,
        help="Sample first N depth snapshots.",
    )
    audit_parser.add_argument(
        "--random-n", type=int, default=10,
        help="Sample random N depth snapshots.",
    )

    debug_parser = subparsers.add_parser("debug-parquet", help="Inspect parquet metadata for one instrument/data type.")
    debug_parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG_ROOT)
    debug_parser.add_argument("--instrument", required=True)
    debug_parser.add_argument("--data-type", required=True, choices=("trade_tick", "order_book_deltas", "order_book_depths"))

    return parser


def serve_command(args: argparse.Namespace) -> int:
    app = create_app(catalog_root=args.catalog, cache_path=args.cache, converter_reports_dir=args.converter_reports, convert_report_date=args.convert_report_date)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0


def audit_command(args: argparse.Namespace) -> int:
    scanner = CatalogScanner(catalog_root=args.catalog, cache_path=args.out, converter_reports_dir=args.converter_reports, convert_report_date=args.convert_report_date)
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
    print(f"avg_l2_quality_score: {s.avg_l2_quality_score:.2f}")
    print(f"avg_audit_confidence_score: {s.avg_audit_confidence_score:.2f}")
    print(f"convert_report_found: {s.convert_report_found}")
    print(f"convert_report_path: {s.convert_report_path or 'none'}")
    print(f"total_fenced_range_count: {s.total_fenced_range_count}")
    print(f"total_desync_count: {s.total_desync_count}")
    print(f"Output: {Path(args.out).expanduser().resolve()}")

    return 0


def debug_parquet_command(args: argparse.Namespace) -> int:
    scanner = CatalogScanner(catalog_root=args.catalog)
    debug = scanner.debug_parquet(instrument_id=args.instrument, data_type=args.data_type)
    print(f"catalog_root: {debug['catalog_root']}")
    print(f"instrument_id: {debug['instrument_id']}")
    print(f"data_type: {debug['data_type']}")
    print("resolved_file_paths:")
    for path in debug["resolved_file_paths"]:
        print(f"  - {path}")
    print("files:")
    for file_info in debug["files"]:
        print(f"  - path: {file_info['path']}")
        print(f"    metadata.num_rows: {file_info['metadata_num_rows']}")
        print(f"    schema_names: {file_info['schema_names']}")
        print(f"    detected_timestamp_column: {file_info['timestamp_column']}")
        print(f"    row_group_stats_available: {file_info['row_group_stats_available']}")
        print(f"    metadata_min_ts: {file_info['metadata_min_ts']}")
        print(f"    metadata_max_ts: {file_info['metadata_max_ts']}")
        print(f"    fallback_min_ts: {file_info['fallback_min_ts']}")
        print(f"    fallback_max_ts: {file_info['fallback_max_ts']}")
        print(f"    timestamp_status: {file_info['timestamp_status']}")
        if file_info["errors"]:
            print(f"    errors: {file_info['errors']}")
    print(f"final_interpreted_status: {debug['final_status']}")
    print(f"timestamp_status: {debug['timestamp_status']}")
    print(f"row_count_estimate: {debug['row_count_estimate']}")
    print(f"row_count_trusted: {debug['row_count_trusted']}")
    print(f"row_count_source: {debug['row_count_source']}")
    print(f"ts_event_min_ns: {debug['ts_event_min_ns']}")
    print(f"ts_event_max_ns: {debug['ts_event_max_ns']}")
    print(f"error: {debug['error']}")
    return 0


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "serve":
        return serve_command(args)
    if args.command == "audit":
        return audit_command(args)
    if args.command == "debug-parquet":
        return debug_parquet_command(args)

    parser.error("Unknown command.")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
