"""Настройки сервиса. Всё читается из переменных окружения."""
import os
from dataclasses import dataclass


def _i(name: str, default: int) -> int:
    return int(os.getenv(name, str(default)))


def _f(name: str, default: float) -> float:
    return float(os.getenv(name, str(default)))


def _b(name: str, default: bool) -> bool:
    return os.getenv(name, "1" if default else "0").strip().lower() in ("1", "true", "yes", "on")


@dataclass(frozen=True)
class Settings:
    db_dsn: str = os.getenv("DB_DSN", "postgresql://case17:case17@db:5432/case17")

    # модель
    embed_model: str = os.getenv("EMBED_MODEL", "resnet18")
    embed_dim: int = _i("EMBED_DIM", 512)
    input_size: int = _i("INPUT_SIZE", 224)
    # Усреднение оригинала и отражения (test-time augmentation): вдвое дороже,
    # даёт +2.17 mAP — замер в research/results.md.
    tta_flip: bool = _b("EMBED_TTA_FLIP", True)
    batch_size: int = _i("BATCH_SIZE", 16)

    # поиск и отказ
    search_top_n: int = _i("SEARCH_TOP_N", 10)
    # Две рабочие точки, обе посчитаны research/operating_points.py на том
    # распределении, в котором работает сервис.
    #   сбалансированная — максимум индекса Юдена, обе ошибки равноценны;
    #   строгая — минимальный порог с TNR >= 0.98: из двадцати чужих запросов
    #             ошибочно принимается не больше одного.
    # Выбор между ними — продуктовое решение, а не техническое, поэтому он
    # вынесен в интерфейс, а не зашит в код.
    search_threshold: float = _f("SEARCH_THRESHOLD", 0.75)
    search_threshold_strict: float = _f("SEARCH_THRESHOLD_STRICT", 0.635)
    search_margin: float = _f("SEARCH_MARGIN", 0.0)

    # загрузка файлов
    max_upload_mb: int = _i("MAX_UPLOAD_MB", 20)
    bbox_pad: float = _f("BBOX_PAD", 0.10)

    data_dir: str = os.getenv("DATA_DIR", "/data")
    artifacts_dir: str = os.getenv("ARTIFACTS_DIR", "/artifacts")

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024


settings = Settings()
