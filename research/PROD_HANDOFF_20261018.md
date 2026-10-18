# PRODUCTION HANDOFF — 2026-10-18 (model track)

Владелец ветки: team17/finals-student-gpu-handoff. Для Глеба/Макса: это модельный отчёт,
production KNN/постобработка/отказ — зона Глеба и Ильи, здесь только данные для интеграции.

## 1. Кандидат модели

Один runtime-студент: **clean_student_a10_ep28.pt**

- файл: `/home/l1zard/diskD/bm_improve/clean_student/clean_student_a10_ep28.pt`
- 379,734,230 bytes (~362 MiB), 94.2M параметров, 512-D, squash 224, fp16-inference, TTA off
- sha256 `93114c5923869d01e92cd70c2c421f783d05ebdac5652a1f2dc919aa040bf6ce`
- метрики (ЛОКАЛЬНЫЙ официальный evaluator-скрипт, а не организаторы):
  - clean holdout 550q/357g (543 scored): mAP@10 **0.8666**, R1 0.8306, R5 0.9466
  - expanded source-domain proxy 550q/8996g (543 scored): mAP@10 **0.5569**, R1 0.5322, R5 0.6814
  - refusal thresholds (локальная калибровка, не prod default): cross-camera 0.6699, no-camera 0.8743

## 2. Код обучения (ТЗ) — на этой ветке

`research/finals_handoff/TEACHER_PIPELINE_README.md` + уже в ветке:
`train_reid.py`, `head_model.py`, `backbones.py`, `extract_emb.py` (tf_mode из чекпоинта),
`build_clean_teacher.py`, `run_clean_retrain.sh`. Полная последовательность команд и
known deviations — в README. `clean_teacher_emb.npy` (8639×512, sha `4ec0e9dc…`).

## 3. Веса — ЗАГРУЖЕНЫ на GitHub LFS (решение команды от 18.10: GitHub, не HF)

Runtime-кандидат в ветке: `weights/clean_student_a10_ep28.pt`
ссылка: https://github.com/makssest06-alt/case-17-vehicle-digital-signature-team-17/raw/team17/finals-student-gpu-handoff/weights/clean_student_a10_ep28.pt
(LFS: 379,734,230 bytes, sha256 `93114c5923869d01e92cd70c2c421f783d05ebdac5652a1f2dc919aa040bf6ce`,
коммит `c4161d4`, `.gitattributes` трекает `*.pt`）

Квоты free GitHub LFS: 1 GB storage / 1 GB bandwidth в месяц. Кандидат занял ~362 MB
storage из 1 GB. Оставшиеся ~640 MB — учителя (~800 MB каждый) НЕ влезают: учителя
остаются локально + Telegram (`/home/l1zard/diskD/bm_improve/clean_models/`, 4 файла +
`ensemble/clean_teacher_emb.npy` sha `4ec0e9dc…`); при необходимости залить одного
SigLIP2-учителя отдельно — предупредить Макса о bandwidth 1 GB/мес.

Ссылку на веса кандидата вставить в README сервиса; при сборке Docker —
`COPY weights/clean_student_a10_ep28.pt artifacts/reid_model.pt` (путь уточнить у Глеба).

## 4. Bench 4060 — полный цикл протокола Q31/Q34 (нативная прокси-оценка)

Важно: `scripts/bench_extract.py` в glebmain отсутствует (проверено на a2606b2 + вся
история веток). После влития PR#5 перезапустить официально:
`docker compose exec api python scripts/bench_extract.py --out /artifacts/bench_extract.json`

Нативные прогоны (артефакты в `research/bench_4060/` совместимой схемы):

| Вариант (TTA / декод) | B1 мс | B32 мс | B32 FPS | гейт ≥100 |
|---|---:|---:|---:|---|
| официально до-правки: TTA on, декод 1 поток | 22.4 | 16.4 | 60.9 | FAIL |
| TTA off, декод 1 поток | 19.4 | 14.5 | 68.8 | FAIL |
| **TTA off, декод 16 threads** | 22.1 | **8.3** | **120.7** | **PASS** |
| TTA on, декод 16 threads | 26.3 | 10.7 | 93.4 | FAIL |

Разбор: узкое место — JPEG-декод 1920×1080 (~11 ms/ядро CPU), НЕ GPU.
GPU forward fp16 ~2.2–4.4 ms. Латентность B1 ≤40 ms выполняется во всех вариантах.
На A5000 + 2×Xeon 6338 (128 thread) ожидается выше 4060-запаса.

### Требуемые default'ы compose для сдачи (Глебу)

1. **TTA off** — качество flip-TTA отмерено как +0.004 mAP (внутри CI нуля), цена −27 FPS.
2. **Батчевый декод в ThreadPoolExecutor(min(16, batch))**: decoding JPEG в потоках.
   При batch==1 последовательный путь быстрее (22.1 против 19.4 ms) — тонкая настройка.
3. FP16 on (уже включён).

## 5. Эксперименты, закрытые отрицательно (не тратить время)

| Эксперимент | Результат |
|---|---|
| EMA (decay .999, paired) | чисто хуже: clean 0.8622 vs 0.8666, expanded 0.5564 vs 0.5569 |
| Cross-batch negative queue 4096 | seed7 +0.0041 (ниже гейта +0.005), seed42 регресс |
| Sub-center ArcFace (2 цент.) | 0.8386 vs 0.8577; испорчено init/несEtherал |
| Late fusion (a10+a25, +teacher) | FAIL: лучший +0.0022 в шуме, учитель-канал вреден |
| Bbox context sweep 0/0.05/0.2/0.35 | 0.10 оптимален, крупнее — падение |
| Flip TTA | +0.004 mAP в шуме, 2× forward — off |
| Учителя-ансамбль отдельно | ensemble 0.8384 — слабше студента a10 (0.8666) |

Эксперимент эльфа в работе (позже): pretrain на VeRi-776 → дистилл alpha10.
VeRi-776 скачан и верифицирован (`/tmp/opencode/veri776/`, 575 train ID, sha `2be9b63b…`).

## 6. Соседняя team 33 — для Глеба (без кода в этой ветке)

`research/finals_handoff/TEAM33_RERANK_REFERENCE.md`: их отображённый вариант
restricted per-query k-reciprocal (k1=6,k2=3,λ=0.3, top-50, без q@q.T) дал +1.7 mAP@10
на их же сплите (0.7242→0.7411). Форма легальна по Q38. Их lenient сплит (750g)
не сравнивается напрямую с нашим clean holdout.

## 7. Остаток до сдачи (роли по состоянию 18.10 вечера)

- [x] код обучения учителей + сборки эмбеддингов (эта ветка, bc29099)
- [x] bench_extract.json на 4060 (нативная прокси + v1/v2/v3)
- [x] веса кандидата на GitHub LFS (c4161d4)
- [ ] официальная bench: docker после PR#5
- [ ] merge model branch в glebmain (Глеб, cherry-pick README+скрипты; reference модель уже в ветке)
- [ ] слайды — Ростик (заготовки с картинками у него)
- [ ] финальная сборка/сдача репо — Макс

Все локальные числа: research/FINAL_WORK_REPORT.md и файлы под research/.
