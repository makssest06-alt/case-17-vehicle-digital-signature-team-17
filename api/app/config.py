"""Настройки сервиса. Всё читается из переменных окружения."""
import os
from dataclasses import dataclass


def _env(name: str, default: str) -> str:
    """Пустая переменная — как незаданная: compose пробрасывает ${X:-} без своего числа."""
    value = os.getenv(name, "")
    return value if value.strip() else default


def _i(name: str, default: int) -> int:
    return int(_env(name, str(default)))


def _f(name: str, default: float) -> float:
    return float(_env(name, str(default)))


def _b(name: str, default: bool) -> bool:
    return os.getenv(name, "1" if default else "0").strip().lower() in ("1", "true", "yes", "on")


@dataclass(frozen=True)
class Settings:
    db_dsn: str = os.getenv("DB_DSN", "postgresql://case17:case17@db:5432/case17")

    # модель
    # Production по умолчанию — clean student alpha=10.  Явный EMBED_MODEL
    # по-прежнему позволяет запускать диагностические профили.
    embed_model: str = os.getenv("EMBED_MODEL", "reid:vit_base_patch16_siglip_224.webli")
    embed_dim: int = _i("EMBED_DIM", 512)
    input_size: int = _i("INPUT_SIZE", 224)
    # Усреднение оригинала и отражения (test-time augmentation): вдвое дороже,
    # даёт +2.17 mAP — замер в research/results.md.
    tta_flip: bool = _b("EMBED_TTA_FLIP", False)
    batch_size: int = _i("BATCH_SIZE", 16)
    # Ожидаемый sha256 файла весов: при старте сверяется, не совпал — предупреждение в логе (app/weights_check.py).
    reid_weights_sha256: str = _env("REID_WEIGHTS_SHA256", "")

    # поиск и отказ
    search_top_n: int = _i("SEARCH_TOP_N", 10)

    # ---- режим отказа: формула оценки и порог — ЗДЕСЬ, одно место (docs/refusal.md) ----
    # Сервис отказывает, если оценка лучшего кандидата меньше SEARCH_THRESHOLD.
    #   raw   — косинус лучшего кандидата (с постобработкой — после неё);
    #   z100  — (s1 − mean(s2..s100)) / std(s2..s100): насколько лучший выделяется
    #           среди ста ближайших; меньше зависит от размера галереи.
    # После выбора постобработки меняется одно число — SEARCH_THRESHOLD (и STRICT
    # для строгого режима); посчитанные значения — REFUSAL_PRESETS ниже.
    refusal_score: str = _env("REFUSAL_SCORE", "z100")
    # Две рабочие точки, обе посчитаны на распределении, в котором работает
    # сервис (галерея не фильтруется по камере: camera_id на вход не подаётся).
    #   сбалансированная — максимум индекса Юдена, обе ошибки равноценны;
    #   строгая — минимальный порог с TNR >= 0.98: из шестидесяти чужих
    #             запросов ошибочно принимается примерно один.
    # Выбор между ними — продуктовое решение, а не техническое, поэтому он
    # вынесен в интерфейс, а не зашит в код.
    search_threshold: float = _f("SEARCH_THRESHOLD", 4.6506)
    search_threshold_strict: float = _f("SEARCH_THRESHOLD_STRICT", 9.6847)
    search_margin: float = _f("SEARCH_MARGIN", 0.0)

    # ---- постобработка: по умолчанию выключена, ответы как без неё (api/app/pp/state.py) ----
    pp_dba_k: int = _i("PP_DBA_K", 1)                                   # production: DBA по 1 соседу
    pp_camera_penalty: str = _env("PP_CAMERA_PENALTY", "on").strip().lower()   # production: признак фона (для отказа)
    pp_camera_tau: float = _f("PP_CAMERA_TAU", 0.68127)   # косинус фона > τ — «та же камера» (cheap_bg_a10.md)
    pp_camera_topk: int = _i("PP_CAMERA_TOPK", 100)       # сколько кандидатов из базы проверять на камеру
    # Исключение почти-дублей: off | bbox (правило рамок: сходство + рамка) | bg (признак фона) | both (OR).
    #   PP_DEDUP_RANK    — какие кандидаты уходят в конец ранжирования;
    #   PP_DEDUP_REFUSAL — какие не учитываются в оценке отказа (пусто — как ранжирование);
    #   PP_DEDUP         — сокращение для обоих (пусто/off — по PP_CAMERA_PENALTY=on → bg).
    # Пороги правила рамок — dedup_compare_a10.md.
    pp_dedup: str = _env("PP_DEDUP", "off").strip().lower()
    pp_dedup_rank: str = _env("PP_DEDUP_RANK", "off").strip().lower()         # production: без флагов (docs/refusal.md)
    pp_dedup_refusal: str = _env("PP_DEDUP_REFUSAL", "both").strip().lower()  # production: фон ИЛИ рамка в отказе
    pp_dedup_sim_any: float = _f("PP_DEDUP_SIM_ANY", float("inf"))   # сырой косинус ≥ — почти-дубль без рамки
    pp_dedup_sim: float = _f("PP_DEDUP_SIM", 0.84)         # сырой косинус ≥ (вместе с рамкой)
    pp_dedup_iou: float = _f("PP_DEDUP_IOU", 0.6)          # IoU рамок в координатах кадра ≥
    pp_dedup_center: float = _f("PP_DEDUP_CENTER", 0.3)    # сдвиг центра в долях среднего размера рамки ≤
    pp_dedup_size: float = _f("PP_DEDUP_SIZE", float("inf"))   # |log| отношения сторон ≤ (inf — не проверять)

    # загрузка файлов
    max_upload_mb: int = _i("MAX_UPLOAD_MB", 20)
    bbox_pad: float = _f("BBOX_PAD", 0.10)

    data_dir: str = os.getenv("DATA_DIR", "/data")
    artifacts_dir: str = os.getenv("ARTIFACTS_DIR", "/artifacts")

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024


# Посчитанные пороги по (PP_DBA_K, исключение в ранжировании, исключение в отказе, формула). Источник — research, команды в
# docs/refusal.md; значения переносятся в SEARCH_THRESHOLD вручную, сервис их не
# подставляет сам: так включённая постобработка без нового порога видна в логе.
REFUSAL_PRESETS = {
    # (PP_DBA_K, исключение в ранжировании, исключение в отказе, REFUSAL_SCORE): {...}
    (0, "off", "off", "raw"): {"balanced": 0.8025, "strict": 0.8465,
                               "note": "legacy: сырой косинус без постобработки (2.2k без почти-дублей, Глеб)"},
    # research/closed_gallery.py --tag a10: 361 / 1111 как закрытый тест, протокол Андрея; итог —
    # (1, "off", "both", "z100"), запасной без фона — (1, "off", "bbox", "z100"); docs/refusal.md
    (1, "bg", "bg", "z100"): {"balanced": 4.995, "strict": 9.9692,
        "note": "DBA + фон: 361/1111 как закрытый тест, F1 при TNR ≥ 0.70 на каждой (closed_gallery_a10.md)"},
    (1, "off", "both", "z100"): {"balanced": 4.6506, "strict": 9.6847,
        "note": "итог: DBA; отказ фон ИЛИ рамка (наши пороги): 361/1111 как закрытый тест, F1 при TNR ≥ 0.70 на каждой (closed_gallery_a10.md)"},
    (1, "bg", "both", "z100"): {"balanced": 4.6506, "strict": 9.6847,
        "note": "прежний итог: DBA + фон; отказ фон ИЛИ рамка (наши пороги): 361/1111 как закрытый тест, F1 при TNR ≥ 0.70 на каждой (closed_gallery_a10.md)"},
    (1, "off", "bbox", "z100"): {"balanced": 4.9039, "strict": 9.9144,
        "note": "запасной: DBA; отказ рамка (наши пороги): 361/1111 как закрытый тест, F1 при TNR ≥ 0.70 на каждой (closed_gallery_a10.md)"},
}


settings = Settings()
