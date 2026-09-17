# Исполнительный план

## День 1

- Data: manifest, SHA-256, image/BBox audit, duplicate checks.
- ML: identity-disjoint split и cross-camera evaluator.
- Retrieval: PyTorch inference, FlatIP, проверка порядка embeddings.npy.
- Backend/MLOps: API skeleton, Docker smoke test, CI.

## День 2: baseline

`ResNet50 -> 512D L2 embedding -> FAISS FlatIP`, CE и optional BatchHardTriplet. Зафиксировать mAP, Rank-1, Rank-5, latency, FPS, RAM/VRAM.

## Дни 3–4: model sweep

ResNet50 CE → ResNet50 CE/Triplet → OSNet CE/Triplet → ConvNeXt-Tiny → crop/context/input size → Color Jitter, Random Erasing, blur, compression, perspective.

## Дни 5–6: hard cases/refusal

- hard-negative mining;
- mask/номер ablation и Grad-CAM;
- calibration refusal;
- error gallery: night, blur, occlusion, viewpoint, similar vehicles.

## День 7: ANN

FlatIP — эталон; HNSW — benchmark; IVF-PQ — только для scale test. Сравнивать recall@1/5/10 относительно FlatIP, latency и RAM. k-reciprocal включать только feature flag и только при устойчивом приросте.

## Дни 8–9: release candidate

ONNX FP32/FP16, output tolerance, weights/index/metadata/configs, три submission artifacts, offline Docker.

## Финальные 24 часа

Code freeze, чистая Linux VM, запуск одной командой, Swagger/UI/health/schema/checksums, финальный tag и архив.
