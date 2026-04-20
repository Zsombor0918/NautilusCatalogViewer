from __future__ import annotations

import hashlib
import json
from collections import OrderedDict
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel


T = TypeVar("T", bound=BaseModel)


def compute_files_signature(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted({Path(path) for path in paths}):
        try:
            stat = path.stat()
            digest.update(str(path).encode("utf-8"))
            digest.update(str(stat.st_mtime_ns).encode("utf-8"))
            digest.update(str(stat.st_size).encode("utf-8"))
        except FileNotFoundError:
            digest.update(str(path).encode("utf-8"))
            digest.update(b"missing")
    return digest.hexdigest()


def build_cache_key(name: str, **parts: object) -> str:
    payload = json.dumps(parts, sort_keys=True, ensure_ascii=False, default=str)
    return f"{name}:{payload}"


class QueryCache:
    def __init__(self, disk_dir: Path | str, max_entries: int = 128) -> None:
        self.disk_dir = Path(disk_dir).expanduser().resolve()
        self.disk_dir.mkdir(parents=True, exist_ok=True)
        self.max_entries = max_entries
        self._memory: OrderedDict[str, tuple[str, str]] = OrderedDict()

    def _cache_file(self, key: str) -> Path:
        name = hashlib.sha256(key.encode("utf-8")).hexdigest()
        return self.disk_dir / f"{name}.json"

    def get_model(self, key: str, signature: str, model_cls: type[T]) -> T | None:
        if key in self._memory:
            cached_signature, cached_payload = self._memory[key]
            if cached_signature == signature:
                self._memory.move_to_end(key)
                return model_cls.model_validate_json(cached_payload)
            self._memory.pop(key, None)

        cache_file = self._cache_file(key)
        if not cache_file.exists():
            return None

        try:
            envelope = json.loads(cache_file.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            cache_file.unlink(missing_ok=True)
            return None

        if envelope.get("signature") != signature:
            cache_file.unlink(missing_ok=True)
            return None

        payload = json.dumps(envelope.get("payload", {}), ensure_ascii=False)
        self._memory[key] = (signature, payload)
        self._memory.move_to_end(key)
        self._trim()
        return model_cls.model_validate(envelope.get("payload", {}))

    def set_model(self, key: str, signature: str, model: BaseModel) -> None:
        payload_json = model.model_dump_json()
        self._memory[key] = (signature, payload_json)
        self._memory.move_to_end(key)
        self._trim()

        envelope = {
            "signature": signature,
            "payload": model.model_dump(mode="json"),
        }
        self._cache_file(key).write_text(
            json.dumps(envelope, ensure_ascii=False, separators=(",", ":")),
            encoding="utf-8",
        )

    def invalidate_prefix(self, prefix: str) -> None:
        for key in list(self._memory):
            if key.startswith(prefix):
                self._memory.pop(key, None)

        for path in self.disk_dir.glob("*.json"):
            try:
                envelope = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                path.unlink(missing_ok=True)
                continue
            payload = envelope.get("payload", {})
            cache_key = payload.get("_cache_key")
            if isinstance(cache_key, str) and cache_key.startswith(prefix):
                path.unlink(missing_ok=True)

    def _trim(self) -> None:
        while len(self._memory) > self.max_entries:
            self._memory.popitem(last=False)
