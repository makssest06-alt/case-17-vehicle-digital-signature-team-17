#!/usr/bin/env python3
"""Сборка трёх артефактов сдачи по требованиям README датасета.

Запускать внутри контейнера api:
    docker compose exec api python scripts/make_submission.py

На выходе в ARTIFACTS_DIR:
  submission.csv   query_id,gallery_id_1..gallery_id_10  — топ-10 по убыванию близости
  embeddings.npy   float32 (N_query + N_gallery, dim); сначала все query
                   в порядке test_query.csv, затем все gallery в порядке test_gallery.csv
  candidates.csv   query_id,gallery_id,confidence — только принятые кандидаты;
                   если для query ничего не прошло порог, строк для него нет (отказ)
"""
import argparse
import csv
import os
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, "/srv")

from app.config import settings          # noqa: E402
from app.models.registry import build_model  # noqa: E402

TOP_K = 10


def read_ids(csv_path: Path) -> list:
    with csv_path.open(newline="") as f:
        return [row["image_id"] for row in csv.DictReader(f)]


def embed_folder(model, folder: Path, ids: list, batch: int, label: str) -> np.ndarray:
    out = np.zeros((len(ids), model.dim), dtype="float32")
    t0 = time.perf_counter()
    for start in range(0, len(ids), batch):
        chunk = ids[start : start + batch]
        imgs = [Image.open(folder / f"{i}.jpg").convert("RGB") for i in chunk]
        out[start : start + len(chunk)] = model.embed_batch(imgs)
        done = start + len(chunk)
        if done % (batch * 10) == 0 or done == len(ids):
            speed = done / (time.perf_counter() - t0)
            print(f"  {label}: {done}/{len(ids)}  {speed:.1f} кадров/с", flush=True)
    return out


def re_ranking(q, g, k1=20, k2=6, lambda_value=0.3):
    """k-reciprocal пересортировка, Zhong et al. 2017.

    Если A нашёл B среди своих k ближайших И B нашёл A среди своих — это
    взаимные соседи, такому совпадению можно верить сильнее одностороннего.
    Дальше считается расстояние Жаккара между множествами взаимных соседей
    и смешивается с исходным косинусным. Меток не требует, ничего не учит.

    Приём транздуктивный: выигрыш берётся из того, что все запросы видны
    одновременно и кадры одной машины подкрепляют друг друга. Поэтому здесь,
    при сборке артефактов, он работает, а в обработчике одиночного запроса —
    нет; замер в research/rerank_online.py.
    """
    nq = len(q)
    total = nq + len(g)
    feat = np.vstack([q, g]).astype(np.float32)

    d = 2.0 - 2.0 * (feat @ feat.T)
    np.maximum(d, 0, out=d)
    d = np.transpose(d / (np.max(d, axis=0) + 1e-12))
    rank = np.argsort(d).astype(np.int32)

    V = np.zeros_like(d, dtype=np.float32)
    half = int(round(k1 / 2.0)) + 1
    for i in range(total):
        fwd = rank[i, : k1 + 1]
        back = rank[fwd, : k1 + 1]
        recip = fwd[np.where(back == i)[0]]
        expanded = recip
        for c in recip:
            c_fwd = rank[c, :half]
            c_back = rank[c_fwd, :half]
            c_recip = c_fwd[np.where(c_back == c)[0]]
            if len(np.intersect1d(c_recip, recip)) > 2.0 / 3 * len(c_recip):
                expanded = np.append(expanded, c_recip)
        expanded = np.unique(expanded)
        w = np.exp(-d[i, expanded])
        V[i, expanded] = w / w.sum()

    if k2 != 1:
        V = np.stack([V[rank[i, :k2]].mean(axis=0) for i in range(total)])

    inv = [np.where(V[:, i] != 0)[0] for i in range(total)]
    jac = np.zeros((nq, total), dtype=np.float32)
    for i in range(nq):
        tmp = np.zeros(total, dtype=np.float32)
        nz = np.where(V[i] != 0)[0]
        for ind in nz:
            rows = inv[ind]
            tmp[rows] += np.minimum(V[i, ind], V[rows, ind])
        jac[i] = 1 - tmp / (2.0 - tmp)

    final = jac * (1 - lambda_value) + d[:nq] * lambda_value
    return -final[:, nq:]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(settings.data_dir, "crops"))
    ap.add_argument("--out", default=settings.artifacts_dir)
    ap.add_argument("--threshold", type=float, default=settings.search_threshold)
    ap.add_argument("--batch", type=int, default=settings.batch_size)
    ap.add_argument("--query-csv", default=None,
                    help="канонический test_query.csv; по умолчанию DATA_DIR/test_query.csv")
    ap.add_argument("--gallery-csv", default=None,
                    help="канонический test_gallery.csv; по умолчанию DATA_DIR/test_gallery.csv")
    ap.add_argument("--rerank", action="store_true",
                    help="k-reciprocal пересортировка выдачи: +3.16 mAP на валидации. "
                         "Меняет только ПОРЯДОК кандидатов; confidence остаётся "
                         "исходным косинусом, чтобы порог отказа не пришлось "
                         "перекалибровывать. Работает только в пакетном режиме — "
                         "выигрыш берётся из того, что весь набор запросов виден "
                         "сразу, поэтому в живом поиске по одному кадру НЕ применяется "
                         "(там он, наоборот, вредит: замер в research/rerank_online.py)")
    ap.add_argument("--rerank-k1", type=int, default=20)
    ap.add_argument("--rerank-k2", type=int, default=6)
    ap.add_argument("--rerank-lambda", type=float, default=0.3)
    ap.add_argument("--reuse-embeddings", action="store_true",
                    help="взять готовый embeddings.npy вместо пересчёта — нужен, когда "
                         "меняется только порог: счёт векторов занимает ~20 минут, "
                         "а порог влияет лишь на candidates.csv")
    args = ap.parse_args()

    data = Path(args.data)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # Порядок строк задаёт README датасета: сначала все image_id из
    # test_query.csv в порядке файла, затем все из test_gallery.csv.
    # Организаторы по embeddings.npy проверяют, что submission.csv получен
    # моделью, а не собран руками, — при другом порядке эта проверка падает.
    # Наши crops_*.csv отсортированы по image_id и каноническому порядку
    # НЕ соответствуют, поэтому читать их здесь нельзя.
    q_csv = Path(args.query_csv) if args.query_csv else Path(settings.data_dir) / "test_query.csv"
    g_csv = Path(args.gallery_csv) if args.gallery_csv else Path(settings.data_dir) / "test_gallery.csv"
    for pth, name in ((q_csv, "test_query.csv"), (g_csv, "test_gallery.csv")):
        if not pth.exists():
            print(f"нет {pth}: без {name} канонический порядок строк восстановить "
                  f"нельзя, а другой порядок организаторы не примут", file=sys.stderr)
            return 1
    q_ids = read_ids(q_csv)
    g_ids = read_ids(g_csv)
    print(f"query: {len(q_ids)}, gallery: {len(g_ids)}")
    if len(g_ids) < TOP_K:
        print(f"ВНИМАНИЕ: в галерее всего {len(g_ids)} объектов, топ-10 не набрать", file=sys.stderr)

    total = len(q_ids) + len(g_ids)
    emb_path = out / "embeddings.npy"

    if args.reuse_embeddings:
        if not emb_path.exists():
            print(f"нет {emb_path} — пересчитать нечего, убери --reuse-embeddings",
                  file=sys.stderr)
            return 1
        saved = np.load(emb_path)
        if saved.shape[0] != total:
            print(f"в {emb_path} {saved.shape[0]} векторов, а нужно {total} — "
                  f"файл от другого набора данных", file=sys.stderr)
            return 1
        # Переиспользовать векторы можно только зная, в каком порядке они легли.
        # Без индекса это ставка на удачу, а порядок здесь — часть контракта.
        idx_path = out / "embeddings_index.csv"
        if not idx_path.exists():
            print(f"нет {idx_path}: порядок строк в {emb_path} неизвестен, "
                  f"переиспользовать нельзя — пересчитай без --reuse-embeddings",
                  file=sys.stderr)
            return 1
        saved_ids = [r["image_id"] for r in csv.DictReader(idx_path.open())]
        want_ids = list(q_ids) + list(g_ids)
        if saved_ids != want_ids:
            if sorted(saved_ids) != sorted(want_ids):
                print("в embeddings.npy другой набор объектов, пересчитай заново",
                      file=sys.stderr)
                return 1
            pos = {v: i for i, v in enumerate(saved_ids)}
            saved = saved[[pos[v] for v in want_ids]]
            np.save(emb_path, saved.astype("float32"))
            with idx_path.open("w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["row", "split", "image_id"])
                for i, v in enumerate(want_ids):
                    w.writerow([i, "query" if i < len(q_ids) else "gallery", v])
            print("порядок строк приведён к каноническому и перезаписан")
        q_emb, g_emb = saved[:len(q_ids)], saved[len(q_ids):]
        dim = saved.shape[1]
        model_name = settings.embed_model
        print(f"векторы взяты из {emb_path}: {saved.shape}, модель не грузится")
    else:
        model = build_model(settings.embed_model, settings.input_size)
        t0 = time.perf_counter()
        model.load()
        print(f"модель {model.name} загружена за {time.perf_counter() - t0:.1f} с, dim={model.dim}")
        dim, model_name = model.dim, model.name

        t_embed = time.perf_counter()
        q_emb = embed_folder(model, data / "query", q_ids, args.batch, "query")
        g_emb = embed_folder(model, data / "gallery", g_ids, args.batch, "gallery")
        embed_s = time.perf_counter() - t_embed
        print(f"эмбеддинги посчитаны за {embed_s:.1f} с — {total / embed_s:.1f} кадров/с")

    # векторы уже L2-нормированы, значит скалярное произведение = косинус
    sim = q_emb @ g_emb.T

    if args.rerank:
        print("ВНИМАНИЕ: --rerank меняет порядок кандидатов, и топ-1 в "
              "submission.csv перестаёт совпадать с топ-1 по embeddings.npy "
              "примерно на 11% запросов. Организаторы сверяют submission с "
              "векторами, поэтому по умолчанию артефакты собираются без "
              "пересортировки. Подробности в docs/SUBMISSION_README.md.",
              file=sys.stderr)
        t_rr = time.perf_counter()
        rr = re_ranking(q_emb, g_emb, args.rerank_k1, args.rerank_k2, args.rerank_lambda)
        order = np.argsort(-rr, axis=1)
        print(f"re-ranking: k1={args.rerank_k1} k2={args.rerank_k2} "
              f"lambda={args.rerank_lambda}, {time.perf_counter() - t_rr:.1f} с")
    else:
        order = np.argsort(-sim, axis=1)

    sub_path = out / "submission.csv"
    with sub_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["query_id"] + [f"gallery_id_{i}" for i in range(1, TOP_K + 1)])
        for qi, q in enumerate(q_ids):
            top = order[qi, :TOP_K]
            w.writerow([q] + [g_ids[j] for j in top])

    index_path = out / "embeddings_index.csv"
    if not args.reuse_embeddings:
        np.save(emb_path, np.vstack([q_emb, g_emb]).astype("float32"))
        # Порядок строк — часть контракта сдачи, поэтому он фиксируется рядом
        # с файлом: так его можно проверить, не пересчитывая векторы.
        with index_path.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["row", "split", "image_id"])
            for i, q in enumerate(q_ids):
                w.writerow([i, "query", q])
            for j, g in enumerate(g_ids):
                w.writerow([len(q_ids) + j, "gallery", g])

    cand_path = out / "candidates.csv"
    accepted, refused = 0, 0
    with cand_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["query_id", "gallery_id", "confidence"])
        for qi, q in enumerate(q_ids):
            rows = [
                (q, g_ids[j], round(float(sim[qi, j]), 4))
                for j in order[qi, :TOP_K]
                if float(sim[qi, j]) >= args.threshold
            ]
            # candidates.csv — это выдача режима отказа, и его контракт —
            # уверенность. При включённом --rerank порядок в order задан
            # пересортировкой, а confidence остаётся исходным косинусом, и
            # список перестаёт убывать. Здесь сортируем по самой уверенности.
            rows.sort(key=lambda r: -r[2])
            if rows:
                w.writerows(rows)
                accepted += 1
            else:
                refused += 1

    print()
    print(f"submission.csv   {sub_path}  строк {len(q_ids)}")
    print(f"embeddings.npy   {emb_path}  форма {(total, dim)}  модель {model_name}")
    print(f"candidates.csv   {cand_path}  порог {args.threshold}")
    print(f"  принято query: {accepted}, отказов: {refused} ({refused / len(q_ids) * 100:.1f}%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
