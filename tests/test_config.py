from __future__ import annotations

from pathlib import Path

from app.config import PROJECT_ROOT, get_viewer_config


def _clear_viewer_env(monkeypatch) -> None:
    for key in (
        "NAUTILUS_CATALOG_ROOT",
        "NAUTILUS_VIEWER_CACHE_PATH",
        "NAUTILUS_VIEWER_CONVERT_REPORT_DIR",
        "NAUTILUS_CONVERTER_REPORTS_DIR",
        "NAUTILUS_VIEWER_CONVERT_REPORT_DATE",
        "NAUTILUS_VIEWER_HOST",
        "NAUTILUS_VIEWER_PORT",
    ):
        monkeypatch.delenv(key, raising=False)


def test_config_file_loads_viewer_defaults(tmp_path: Path, monkeypatch) -> None:
    _clear_viewer_env(monkeypatch)
    config_path = tmp_path / "viewer.env"
    config_path.write_text(
        "\n".join(
            [
                "NAUTILUS_CATALOG_ROOT=~/sync/catalog",
                "NAUTILUS_VIEWER_CACHE_PATH=state/custom_cache.json",
                "NAUTILUS_VIEWER_CONVERT_REPORT_DIR=~/sync/convert_reports",
                "NAUTILUS_VIEWER_HOST=0.0.0.0",
                "NAUTILUS_VIEWER_PORT=8123",
            ],
        ),
        encoding="utf-8",
    )

    config = get_viewer_config(config_path)

    assert config.catalog_root == (Path.home() / "sync" / "catalog").resolve()
    assert config.cache_path == (PROJECT_ROOT / "state/custom_cache.json").resolve()
    assert config.converter_reports_dir == (Path.home() / "sync" / "convert_reports").resolve()
    assert config.host == "0.0.0.0"
    assert config.port == 8123


def test_environment_overrides_config_file(tmp_path: Path, monkeypatch) -> None:
    _clear_viewer_env(monkeypatch)
    config_path = tmp_path / "viewer.env"
    config_path.write_text("NAUTILUS_CATALOG_ROOT=/from/file\n", encoding="utf-8")
    monkeypatch.setenv("NAUTILUS_CATALOG_ROOT", str(tmp_path / "from-env"))

    config = get_viewer_config(config_path)

    assert config.catalog_root == (tmp_path / "from-env").resolve()
