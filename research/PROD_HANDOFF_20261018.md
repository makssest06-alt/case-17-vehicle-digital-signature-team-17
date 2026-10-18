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

## 4. Bench 4060 — ОФИЦИАЛЬНЫЙ код (PR#5 слит, эффект от `api/scripts/bench_extract.py`)

После мержа PR#5 (glebmain e41f92a, 18.10) официальный скрипт запущен нативно на 4060
(контейнерный путь блокирован отсутствием sudo у nvidia-container-toolkit; скрипт/код официальные).

| EXTRACT_THREADS | B1 мс (median) | FPS 1/8/16/32 | best | perf баллов (из 20) |
|---:|---:|---|---:|---:|
| 1 | 19.65 | 59.7 / 58.7 / 62.4 / 61.3 | 62.4 | 12.48 |
| **4 (дефолт)** | **20.07** | **56.1 / 83.5 / 89.1 / 94.4** | **94.4** | **18.87** |
| 16 | 20.28 | 57.2 / 80.1 / 93.8 / 94.1 | 94.1 | 18.81 |

B1 гейт закрыт: native 20.07 мс (≤40, запас 2×); container 22.0 мс. Best FPS на 4060 Laptop:
native 94.4 / container 91.2 (perf 18.87 / 18.24 из 20 в контейнерном дефолте). VRAM 533 / 698 МБ,
детерминизм 0, веса 362 МБ без сети. Потоки (t8/t16) и вне, и в контейнере выше 92–94 FPS не дают —
узкое место: декод/препроцесс CPU ноутбука; на A5000 + 64-ядерном Xeon ожидается ≥100 при дефолтах.
Подробности + файлы JSON (native + docker): `research/bench_4060_official/`.

### Дейфолты для сдачи — уже в слитом PR#5: FP16 on, TTA off, EXTRACT_THREADS=4, JPEG draft off.
Контейнерное воспроизведение готово и отработано на 4060 (nvidia-container-toolkit установлен,
`docker compose build api` = 247 с, задача в README §5 запускается без правок). Осталось повторить
на A5000-машине (владелец панели) для финального артефакта bench_extract.json с той машины.

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
- [x] bench_extract.json на 4060 — официальный код PR#5 (t1/t4/t16; контейнер — только с sudo)
- [x] веса кандидата на GitHub LFS (c4161d4)
- [ ] bench на A5000 в claim-подтверждении (кто владеет A5000)
- [x] merge model branch в glebmain: уч. PR#5 уже перенис README «Обучение: как воспроизвести a10»
      (кроме двух шагов: SigLIP2-учитель + сборка teacher_emb — они в моих обновлениях §2/проверить)
- [x] слайды — Ростик (заготовки с картинками у него)
- [ ] финальная сборка/сдача репо — Макс

Все локальные числа: research/FINAL_WORK_REPORT.md и файлы под research/.
