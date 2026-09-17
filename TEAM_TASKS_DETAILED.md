# Задачи для пяти участников

## 1. ML Lead

Владелец split, evaluator, cross-camera metrics, refusal calibration и выбора финального checkpoint. Решает, что обучать.

## 2. Data/GPU Engineer

Владелец dataset audit, BBox/crops, dataloader, sampler, augmentations, Yandex Cloud/арендованных GPU, checkpoints, логов и стоимости.

## 3. Model/Retrieval Engineer

Владелец ResNet/OSNet/ConvNeXt adapters, embedding export, FlatIP/HNSW/IVF-PQ, reranking, inference и трёх submission artifacts.

## 4. Backend/Demo Engineer

Владелец FastAPI/OpenAPI, upload image+BBox, top-k/confidence/refusal, UI и debug Grad-CAM endpoint.

## 5. MLOps/Docs/Presentation

Владелец CI, Docker, offline smoke test, README, Model Card, лицензий, error analysis, слайдов 7–11 и финального release.

## Границы решений

- что обучать и как оценивать — ML Lead;
- как быстро исполнять — Model/Retrieval;
- где и как обучать — Data/GPU;
- как вызывать сервис — Backend;
- как воспроизводить и защищать — MLOps/Docs.
