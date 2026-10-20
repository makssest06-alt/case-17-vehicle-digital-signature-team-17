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
    # Production — soup150s2: model soup студентов ViT-L/16@256 l150 (seed 7) и s2 (seed 42), research/make_soup.py;
    # clean mAP@10 0.9133, expanded 0.6592 (l150: 0.9005 / 0.6467; запасная a10: 0.8666 / 0.5569; manifest.json).
    # Явный EMBED_MODEL по-прежнему позволяет запускать диагностические профили.
    embed_model: str = os.getenv("EMBED_MODEL", "reid:vit_large_patch16_siglip_256")
    embed_dim: int = _i("EMBED_DIM", 512)
    input_size: int = _i("INPUT_SIZE", 256)
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
    # Production — raw: на протоколе по ответу Q17 оценка организаторов 0.7·F1 + 0.3·TNR у raw выше и с
    # порогом по своей галерее, и при переносе 361 <-> 1111; z100 — пресет по выбору (docs/refusal.md).
    # Отказ по умолчанию — без исключения почти-дублей (пресет «DBA», с 20.10); на протоколе Q17 у soup150s2
    # он лучше прежнего итога: 0.943 / 0.893 против 0.896 / 0.789 (recalibrate_soup150s2.json); прежний итог — запасной.
    # После выбора постобработки меняется одно число — SEARCH_THRESHOLD (и STRICT
    # для строгого режима); посчитанные значения — REFUSAL_PRESETS ниже.
    refusal_score: str = _env("REFUSAL_SCORE", "raw")
    # Две рабочие точки, обе посчитаны на распределении, в котором работает
    # сервис (галерея не фильтруется по камере: camera_id на вход не подаётся).
    #   сбалансированная — максимум оценки организаторов 0.7·F1 + 0.3·TNR (Q15);
    #   строгая — наименьший порог с TNR >= 0.98 на обеих галереях на фолде выбора;
    #             вне фолда TNR 0.991 на 361 и 0.964 на 1111 (recalibrate_soup150s2.json).
    # Выбор между ними — продуктовое решение, а не техническое, поэтому он
    # вынесен в интерфейс, а не зашит в код.
    search_threshold: float = _f("SEARCH_THRESHOLD", 0.8881)
    search_threshold_strict: float = _f("SEARCH_THRESHOLD_STRICT", 0.9445)
    search_margin: float = _f("SEARCH_MARGIN", 0.0)

    # ---- постобработка: по умолчанию выключена, ответы как без неё (api/app/pp/state.py) ----
    pp_dba_k: int = _i("PP_DBA_K", 1)                                   # production: DBA по 1 соседу
    pp_camera_penalty: str = _env("PP_CAMERA_PENALTY", "on").strip().lower()   # признак фона считается и хранится; в отказе — у запасного пресета
    pp_camera_tau: float = _f("PP_CAMERA_TAU", 0.68233)   # косинус фона > τ — «та же камера» (recalibrate_soup150s2.json)
    pp_camera_topk: int = _i("PP_CAMERA_TOPK", 100)       # сколько кандидатов из базы проверять на камеру
    # Исключение почти-дублей: off | bbox (правило рамок: сходство + рамка) | bg (признак фона) | both (OR).
    #   PP_DEDUP_RANK    — какие кандидаты уходят в конец ранжирования;
    #   PP_DEDUP_REFUSAL — какие не учитываются в оценке отказа (пусто — как ранжирование);
    #   PP_DEDUP         — сокращение для обоих (пусто/off — по PP_CAMERA_PENALTY=on → bg).
    # Пороги правила рамок — recalibrate_soup150s2.json (research/recalibrate_for_model.py --tag soup150s2).
    pp_dedup: str = _env("PP_DEDUP", "off").strip().lower()
    pp_dedup_rank: str = _env("PP_DEDUP_RANK", "off").strip().lower()         # production: без флагов (docs/refusal.md)
    pp_dedup_refusal: str = _env("PP_DEDUP_REFUSAL", "off").strip().lower()   # production: без исключения (both — запасной)
    pp_dedup_sim_any: float = _f("PP_DEDUP_SIM_ANY", float("inf"))   # сырой косинус ≥ — почти-дубль без рамки
    pp_dedup_sim: float = _f("PP_DEDUP_SIM", 0.87)         # сырой косинус ≥ (вместе с рамкой)
    pp_dedup_iou: float = _f("PP_DEDUP_IOU", 0.6)          # IoU рамок в координатах кадра ≥
    pp_dedup_center: float = _f("PP_DEDUP_CENTER", 0.3)    # сдвиг центра в долях среднего размера рамки ≤
    pp_dedup_size: float = _f("PP_DEDUP_SIZE", float("inf"))  # |log| отношения сторон ≤ (inf — не проверять)

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
                               "note": "legacy: сырой косинус без постобработки (2.2k без почти-дублей, Глеб; калибровано под a10)"},
    # research/closed_gallery.py --tag soup150s2 через research/recalibrate_for_model.py --tag soup150s2
    # (протокол q17 — как в ответе Q17): 361 и 361 + 750 кадров обучающих машин, без кадров теста;
    # у 20 % машин в галерее нет ни одного кадра; максимум 0.7·F1 + 0.3·TNR (Q15). Итог — (1, "off", "off", "raw"),
    # «DBA» без исключения почти-дублей (presets["DBA"] того же прогона; в config_presets его пишет новый
    # генератор); запасные — (1, "off", "both", "raw") (прежний итог, отказ фон ИЛИ рамка) и (1, "off", "bbox", "raw")
    # без фона; z100 — те же ключи с "z100". research/honest_eval_results/recalibrate_soup150s2.json, closed_gallery_soup150s2.json.
    (1, "off", "off", "raw"): {"balanced": 0.8881, "strict": 0.9445,
        "note": "итог: DBA, отказ без исключения почти-дублей, raw: 361 и 1111 (+750 кадров обучающих машин), без кадров теста, 20 % машин без пары (Q17); 0.7·F1 + 0.3·TNR (closed_gallery_soup150s2.json)"},
    (1, "off", "both", "raw"): {"balanced": 0.8195, "strict": 0.9443,
        "note": "итог: DBA; отказ фон ИЛИ рамка (наши пороги), raw: 361 и 1111 (+750 кадров обучающих машин), без кадров теста, 20 % машин без пары (Q17); 0.7·F1 + 0.3·TNR (recalibrate_soup150s2.json)"},
    (1, "off", "bbox", "raw"): {"balanced": 0.8244, "strict": 0.9442,
        "note": "запасной: DBA; отказ рамка (наши пороги), raw: 361 и 1111 (+750 кадров обучающих машин), без кадров теста, 20 % машин без пары (Q17); 0.7·F1 + 0.3·TNR (recalibrate_soup150s2.json)"},
    (1, "off", "off", "z100"): {"balanced": 4.2787, "strict": 8.698,
        "note": "итог: DBA, отказ без исключения почти-дублей, z100: 361 и 1111 (+750 кадров обучающих машин), без кадров теста, 20 % машин без пары (Q17); 0.7·F1 + 0.3·TNR (closed_gallery_soup150s2.json)"},
    (1, "off", "both", "z100"): {"balanced": 4.3777, "strict": 8.5204,
        "note": "итог: DBA; отказ фон ИЛИ рамка (наши пороги), z100: 361 и 1111 (+750 кадров обучающих машин), без кадров теста, 20 % машин без пары (Q17); 0.7·F1 + 0.3·TNR (recalibrate_soup150s2.json)"},
    (1, "off", "bbox", "z100"): {"balanced": 4.4228, "strict": 8.5754,
        "note": "запасной: DBA; отказ рамка (наши пороги), z100: 361 и 1111 (+750 кадров обучающих машин), без кадров теста, 20 % машин без пары (Q17); 0.7·F1 + 0.3·TNR (recalibrate_soup150s2.json)"},
}


settings = Settings()
