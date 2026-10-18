# ---- веса clean_student_a10_ep28.pt: без git-lfs у того, кто собирает ----
# weights_ctx — каталог weights/ репозитория, docker-compose.yml передаёт его вторым
# контекстом сборки (additional_contexts). Без него (docker build api) стадия пустая,
# и веса берутся только по WEIGHTS_URL.
FROM scratch AS weights_ctx

FROM python:3.11-slim AS weights
# Настоящий файл в weights/ — копируется, ничего не качается. LFS-указатель или файла
# нет — скачивается по WEIGHTS_URL. sha256 сверяется всегда (scripts/fetch_weights.sh).
# WEIGHTS_URL пуст: где публиковать веса, решает команда; до тех пор — git lfs pull.
ARG WEIGHTS_URL=""
ARG WEIGHTS_SHA256=93114c5923869d01e92cd70c2c421f783d05ebdac5652a1f2dc919aa040bf6ce
COPY scripts/fetch_weights.sh /usr/local/bin/fetch_weights.sh
RUN --mount=type=bind,from=weights_ctx,target=/ctx \
    sh /usr/local/bin/fetch_weights.sh /ctx/clean_student_a10_ep28.pt \
       /opt/weights/clean_student_a10_ep28.pt "$WEIGHTS_URL" "$WEIGHTS_SHA256"

FROM python:3.11-slim

ARG EMBED_MODEL=reid:vit_base_patch16_siglip_224.webli
ENV EMBED_MODEL=${EMBED_MODEL}
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/opt/hfcache \
    OMP_NUM_THREADS=2

WORKDIR /srv

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
 && python -c "import torch; assert torch.version.cuda, 'CUDA torch build required'; print('torch', torch.__version__, 'cuda', torch.version.cuda)"

# Веса скачиваем на этапе сборки и запекаем в образ: в рантайме интернета нет
# EMBED_MODEL может быть вида reid:<бэкбон> или timm:<бэкбон> — для запекания
# весов нужен сам бэкбон, поэтому префикс отбрасываем.
RUN python -c "\
import timm, os; \
name = os.environ.get('EMBED_MODEL', '${EMBED_MODEL}'); \
name = name.split(':', 1)[1] if ':' in name else name; \
m = timm.create_model(name, pretrained=True, num_classes=0); \
print('весов запечено:', name, m.num_features)" 

# Веса модели — свой путь в образе, не под /models: том ./weights:/models из
# docker-compose.yml их не перекрывает и для запуска не нужен.
COPY --from=weights /opt/weights/ /opt/weights/
ENV REID_WEIGHTS=/opt/weights/clean_student_a10_ep28.pt

COPY app ./app
COPY scripts ./scripts

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
