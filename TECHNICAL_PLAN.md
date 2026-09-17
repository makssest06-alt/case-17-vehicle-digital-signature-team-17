# Технический план

## 1. Data contract

Сделать `validate_data.py`, `dataset_report.json`, `manifest.csv`, `split_v1.json`. Проверить image IDs, BBox, разрешение 1920×1080, дубликаты, identity leakage и camera distribution.

## 2. Baseline

```text
BBox crop → ResNet50 → 512D → L2 → FAISS FlatIP → evaluator
```

Снять mAP/Rank-1/Rank-5, latency batch=1, FPS, RAM/VRAM и размер весов.

## 3. Model sweep

В фиксированном validation split сравнить ResNet50 CE, ResNet50 CE/Triplet, OSNet CE/Triplet, ConvNeXt-Tiny, crop/context, input sizes и realistic augmentations. Каждое изменение — отдельный experiment ID.

## 4. Retrieval

FlatIP — эталон качества на gallery=750. HNSW и IVF-PQ — benchmark масштабирования до 100k/1M synthetic vectors. k-reciprocal — отдельный flag с оценкой mAP и latency.

## 5. Refusal

Собрать known-in-gallery и unknown-to-gallery validation, построить PR curve, подобрать calibrated threshold по F1/TNR, заморозить operating point и описать в README.

## 6. Export/API/release

PyTorch reference → ONNX FP32/FP16, tolerance test, FastAPI OpenAPI, model/index load once, health/readiness, Docker offline run, генерация `submission.csv`, `embeddings.npy`, `candidates.csv`, checksum и release tag.
