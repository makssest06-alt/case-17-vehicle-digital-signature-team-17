"""sha256 файла весов при старте: записать в лог и сверить с ожидаемым из конфига.

Считается один раз в lifespan; /health отдаёт его полем weights_sha256 (без файла весов поля нет,
ответ прежний). Остальные ответы не меняются, при несовпадении — только предупреждение в логе.
Ожидаемое значение — REID_WEIGHTS_SHA256; пусто — только записать. Для a10:
93114c5923869d01e92cd70c2c421f783d05ebdac5652a1f2dc919aa040bf6ce
(research/clean_student/manifest.json ветки team17/finals-student-gpu-handoff).
"""
import hashlib
import logging
import os
from pathlib import Path
from typing import Optional, Tuple

log = logging.getLogger("case17")


def weights_path(embed_model: str) -> Optional[Path]:
    """Файл весов модели: REID_WEIGHTS для reid:<бэкбон>; у timm-заглушки файла нет."""
    if embed_model.startswith("reid:"):
        return Path(os.getenv("REID_WEIGHTS", "/opt/weights/soup150s2.pt"))
    return None


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def inspect(embed_model: str, expected: str) -> Tuple[Optional[str], Optional[bool]]:
    """-> (sha256 файла или None без файла; True — совпал, False — не совпал, None — сверять не с чем)."""
    path = weights_path(embed_model)
    if path is None or not path.is_file():
        return None, None
    sha = sha256_file(path)
    want = (expected or "").strip().lower()
    if not want:
        log.info("веса %s: sha256 %s (ожидаемый не задан, REID_WEIGHTS_SHA256)", path, sha)
        return sha, None
    if sha == want:
        log.info("веса %s: sha256 %s — совпадает с REID_WEIGHTS_SHA256", path, sha)
        return sha, True
    log.warning("веса %s: sha256 %s, а ожидается %s (REID_WEIGHTS_SHA256) — не те веса?", path, sha, want)
    return sha, False


def check(embed_model: str, expected: str) -> Optional[bool]:
    """Только итог сверки (см. inspect)."""
    return inspect(embed_model, expected)[1]
