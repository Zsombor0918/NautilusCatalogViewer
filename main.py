from __future__ import annotations

import argparse
from pathlib import Path

import uvicorn

from app.config import DEFAULT_CONFIG_PATH, get_viewer_config
from app.main import create_app


def _path_arg(value: str | None) -> Path | None:
    if not value:
        return None
    return Path(value).expanduser().resolve()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Launch the Nautilus Catalog Viewer.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH, help="Path to viewer env config.")
    parser.add_argument("--host", default=None, help="Bind host. Defaults to config value.")
    parser.add_argument("--port", type=int, default=None, help="Bind port. Defaults to config value.")
    parser.add_argument("--catalog", default=None, help="Catalog root override.")
    parser.add_argument("--cache", default=None, help="Audit cache path override.")
    parser.add_argument("--convert-report-dir", "--converter-reports", default=None, dest="converter_reports", help="Convert reports directory override.")
    parser.add_argument("--convert-report-date", default=None, help="Convert report date override, e.g. 2026-05-15.")
    return parser


app = create_app()


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    config = get_viewer_config(args.config)

    server_app = create_app(
        catalog_root=_path_arg(args.catalog) or config.catalog_root,
        cache_path=_path_arg(args.cache) or config.cache_path,
        converter_reports_dir=_path_arg(args.converter_reports) or config.converter_reports_dir,
        convert_report_date=args.convert_report_date or config.convert_report_date,
    )
    uvicorn.run(server_app, host=args.host or config.host, port=args.port or config.port, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
