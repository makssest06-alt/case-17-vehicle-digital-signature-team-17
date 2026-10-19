# Контекст сборки — корень репозитория (docker-compose.yml: context: ., dockerfile: api/Dockerfile);
# корневой .dockerignore — белый список: api/ и weights/clean_student_large_150_ep150.pt.

# ---- веса clean_student_large_150_ep150.pt: скачиваются при сборке (в git их нет — 1.21 ГБ) ----
FROM python:3.11.17-slim AS weights
# Настоящий файл из weights/ остаётся как есть, ничего не качается. Файла нет
# (шаблон weight[s]/… тогда ничего не копирует) — скачивается по WEIGHTS_URL.
# sha256 сверяется всегда (api/scripts/fetch_weights.sh).
# WEIGHTS_URL по умолчанию — ассет GitHub Release weights-l150 (публичный, без LFS-квоты).
ARG WEIGHTS_URL="https://github.com/makssest06-alt/case-17-vehicle-digital-signature-team-17/releases/download/weights-l150/clean_student_large_150_ep150.pt"
ARG WEIGHTS_SHA256=6937d1163e239bcfd8ee76fe00da12a9d60becc89563b8fc5a475dc72df13d52
COPY weight[s]/clean_student_large_150_ep150.p[t] /opt/weights/
COPY api/scripts/fetch_weights.sh /usr/local/bin/fetch_weights.sh
RUN sh /usr/local/bin/fetch_weights.sh /opt/weights/clean_student_large_150_ep150.pt \
       /opt/weights/clean_student_large_150_ep150.pt "$WEIGHTS_URL" "$WEIGHTS_SHA256"

FROM python:3.11.17-slim

ARG EMBED_MODEL=reid:vit_large_patch16_siglip_256
ARG BAKE_BACKBONE=0
ENV EMBED_MODEL=${EMBED_MODEL}
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/opt/hfcache \
    HF_HUB_OFFLINE=1 \
    REID_TF_MODE=squash \
    OMP_NUM_THREADS=2

WORKDIR /srv

COPY api/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
 && python -c "import torch; assert torch.version.cuda, 'CUDA torch build required'; print('torch', torch.__version__, 'cuda', torch.version.cuda)"

# Предобученный бэкбон с HF не нужен: чекпоинт полного дообучения несёт весь энкодер
# (pretrained=False), без сети и без HF-кэша модель грузится — проверено. Запекание
# (+1.2 ГБ к весам по Q37) — только для timm:<бэкбон> (zero-shot) и head-only reid:-чекпоинтов:
#   docker compose build --build-arg BAKE_BACKBONE=1   (или BAKE_BACKBONE=1 в .env)
# Тогда скачивается последняя ревизия HF-репозитория бэкбона — по Q39 её надо закрепить отдельно.
RUN if [ "$BAKE_BACKBONE" = "1" ]; then HF_HUB_OFFLINE=0 python -c "\
import timm, os; \
name = os.environ.get('EMBED_MODEL', '${EMBED_MODEL}'); \
name = name.split(':', 1)[1] if ':' in name else name; \
m = timm.create_model(name, pretrained=True, num_classes=0); \
print('весов запечено:', name, m.num_features)"; else echo "бэкбон не запекается (BAKE_BACKBONE=0)"; fi

# Веса модели — свой путь в образе, не под /models: том ./weights:/models из
# docker-compose.yml их не перекрывает и для запуска не нужен.
COPY --from=weights /opt/weights/ /opt/weights/
ENV REID_WEIGHTS=/opt/weights/clean_student_large_150_ep150.pt
# REID_TF_MODE=squash и HF_HUB_OFFLINE=1 — в ENV выше: docker run без compose (make_submission, bench_extract)
# работает так же, как сервис.

COPY api/app ./app
COPY api/scripts ./scripts

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
