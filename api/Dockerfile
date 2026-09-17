FROM python:3.11-slim

ARG EMBED_MODEL=resnet18
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/opt/hfcache \
    OMP_NUM_THREADS=2

WORKDIR /srv

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
 && python -c "import torch; assert not torch.version.cuda, 'в образ попала CUDA-сборка torch'; print('torch', torch.__version__)"

# Веса скачиваем на этапе сборки и запекаем в образ: в рантайме интернета нет
RUN python -c "import timm; m = timm.create_model('${EMBED_MODEL}', pretrained=True, num_classes=0); print('весов запечено, признаков:', m.num_features)"

COPY app ./app
COPY scripts ./scripts

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
