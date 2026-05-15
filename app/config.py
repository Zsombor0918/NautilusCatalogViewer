from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = Path(
    os.getenv("NAUTILUS_VIEWER_CONFIG", str(PROJECT_ROOT / "config" / "viewer.env")),
).expanduser()


@dataclass(frozen=True)
class ViewerConfig:
    catalog_root: Path
    cache_path: Path
    converter_reports_dir: Path | None
    convert_report_date: str | None
    host: str
    port: int


def _read_env_file(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}

    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line.removeprefix("export ").strip()
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        values[key] = os.path.expandvars(value)
    return values


def _configured_value(file_values: dict[str, str], key: str, default: str | None = None) -> str | None:
    env_value = os.getenv(key)
    if env_value:
        return env_value
    file_value = file_values.get(key)
    if file_value:
        return file_value
    return default


def _path_value(value: str | None, default: Path) -> Path:
    path = Path(value).expanduser() if value else default.expanduser()
    if path.is_absolute():
        return path.resolve()
    return (PROJECT_ROOT / path).resolve()


def _optional_path_value(value: str | None) -> Path | None:
    if not value:
        return None
    return _path_value(value, PROJECT_ROOT)


def _int_value(value: str | None, default: int) -> int:
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def get_viewer_config(config_path: Path | str | None = None) -> ViewerConfig:
    path = Path(config_path).expanduser() if config_path else DEFAULT_CONFIG_PATH
    file_values = _read_env_file(path)

    catalog_root = _path_value(
        _configured_value(file_values, "NAUTILUS_CATALOG_ROOT"),
        Path.home() / "sync" / "catalog",
    )
    cache_path = _path_value(
        _configured_value(file_values, "NAUTILUS_VIEWER_CACHE_PATH"),
        PROJECT_ROOT / "state" / "web_audit_cache.json",
    )
    converter_reports_dir = _optional_path_value(
        _configured_value(file_values, "NAUTILUS_VIEWER_CONVERT_REPORT_DIR")
        or _configured_value(file_values, "NAUTILUS_CONVERTER_REPORTS_DIR"),
    )
    convert_report_date = _configured_value(file_values, "NAUTILUS_VIEWER_CONVERT_REPORT_DATE")
    host = _configured_value(file_values, "NAUTILUS_VIEWER_HOST", "127.0.0.1") or "127.0.0.1"
    port = _int_value(_configured_value(file_values, "NAUTILUS_VIEWER_PORT"), 8000)

    return ViewerConfig(
        catalog_root=catalog_root,
        cache_path=cache_path,
        converter_reports_dir=converter_reports_dir,
        convert_report_date=convert_report_date,
        host=host,
        port=port,
    )


DEFAULT_VIEWER_CONFIG = get_viewer_config()
DEFAULT_CATALOG_ROOT = DEFAULT_VIEWER_CONFIG.catalog_root
DEFAULT_CACHE_PATH = DEFAULT_VIEWER_CONFIG.cache_path
DEFAULT_CONVERTER_REPORTS_DIR = DEFAULT_VIEWER_CONFIG.converter_reports_dir
DEFAULT_CONVERT_REPORT_DATE = DEFAULT_VIEWER_CONFIG.convert_report_date
DEFAULT_HOST = DEFAULT_VIEWER_CONFIG.host
DEFAULT_PORT = DEFAULT_VIEWER_CONFIG.port
