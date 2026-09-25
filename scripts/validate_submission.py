#!/usr/bin/env python3
"""Формальная проверка артефактов сдачи против README датасета.

    python scripts/validate_submission.py
    python scripts/validate_submission.py --artifacts artifacts --data /opt/case17/data
    python scripts/validate_submission.py --strict     # предупреждения тоже роняют

Код возврата 0 — артефакты можно коммитить. Любой другой — нельзя.

Проверяется ровно то, что описано в README датасета, плюс то, на чём легко
обжечься молча:

  submission.csv   заголовок, число строк, порядок query_id как в
                   test_query.csv, ровно 10 кандидатов, все id существуют,
                   внутри строки нет повторов;
  embeddings.npy   float32, форма (N_query + N_gallery, dim), порядок строк
                   канонический, нет NaN, нормы единичные, нет дублей;
  candidates.csv   заголовок, id существуют, confidence числовой и убывает,
                   нет пар-дублей, отказы посчитаны;
  сквозная         топ-1 из embeddings.npy совпадает с первой колонкой
                   submission.csv. Организаторы делают ровно эту проверку,
                   чтобы отличить работу модели от собранной руками таблицы.
"""
import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

TOP_K = 10
SUB_HEADER = ["query_id"] + [f"gallery_id_{i}" for i in range(1, TOP_K + 1)]
CAND_HEADER = ["query_id", "gallery_id", "confidence"]


class Report:
    def __init__(self):
        self.fails, self.warns, self.oks = [], [], []

    def ok(self, msg):
        self.oks.append(msg)
        print(f"  \033[32mOK\033[0m   {msg}")

    def warn(self, msg):
        self.warns.append(msg)
        print(f"  \033[33mWARN\033[0m {msg}")

    def fail(self, msg):
        self.fails.append(msg)
        print(f"  \033[31mFAIL\033[0m {msg}")


def section(title):
    print(f"\n\033[1m{title}\033[0m")


def read_ids(path: Path):
    with path.open() as f:
        return [r["image_id"] for r in csv.DictReader(f)]


def check_submission(art: Path, q_ids, g_ids, rep: Report):
    section("submission.csv")
    p = art / "submission.csv"
    if not p.exists():
        rep.fail(f"нет файла {p}")
        return None

    with p.open() as f:
        rows = list(csv.reader(f))
    if not rows:
        rep.fail("файл пуст")
        return None

    if rows[0] != SUB_HEADER:
        rep.fail(f"заголовок не по README: {rows[0][:3]}... ожидался {SUB_HEADER[:3]}...")
    else:
        rep.ok(f"заголовок корректный ({len(SUB_HEADER)} колонок)")

    body = rows[1:]
    if len(body) != len(q_ids):
        rep.fail(f"строк {len(body)}, а query в test_query.csv {len(q_ids)}")
    else:
        rep.ok(f"строк {len(body)} — совпадает с test_query.csv")

    got_q = [r[0] for r in body]
    if got_q != q_ids:
        if sorted(got_q) == sorted(q_ids):
            rep.fail("порядок query_id не совпадает с test_query.csv "
                     "(набор тот же, последовательность другая)")
        else:
            rep.fail("набор query_id не совпадает с test_query.csv")
    else:
        rep.ok("порядок query_id совпадает с test_query.csv")

    if len(set(got_q)) != len(got_q):
        rep.fail(f"дубли query_id: {len(got_q) - len(set(got_q))} шт")

    gset = set(g_ids)
    bad_width = sum(1 for r in body if len(r) != TOP_K + 1)
    if bad_width:
        rep.fail(f"строк с числом кандидатов не {TOP_K}: {bad_width}")
    else:
        rep.ok(f"в каждой строке ровно {TOP_K} кандидатов")

    unknown = {c for r in body for c in r[1:] if c and c not in gset}
    if unknown:
        rep.fail(f"кандидатов нет в test_gallery.csv: {len(unknown)} шт, "
                 f"например {sorted(unknown)[0]}")
    else:
        rep.ok("все кандидаты есть в test_gallery.csv")

    empty = sum(1 for r in body for c in r[1:] if not c)
    if empty:
        rep.fail(f"пустых ячеек кандидатов: {empty}")

    dupes = sum(1 for r in body if len(set(r[1:])) != len(r[1:]))
    if dupes:
        rep.fail(f"строк с повторяющимся кандидатом внутри строки: {dupes}")
    else:
        rep.ok("внутри строк повторов нет")

    return {r[0]: r[1:] for r in body}


def check_embeddings(art: Path, q_ids, g_ids, rep: Report):
    section("embeddings.npy")
    p = art / "embeddings.npy"
    if not p.exists():
        rep.fail(f"нет файла {p}")
        return None, None

    emb = np.load(p)
    total = len(q_ids) + len(g_ids)
    if emb.dtype != np.float32:
        rep.warn(f"dtype {emb.dtype}, README подразумевает float32")
    else:
        rep.ok("dtype float32")

    if emb.ndim != 2 or emb.shape[0] != total:
        rep.fail(f"форма {emb.shape}, ожидалась ({total}, dim)")
        return emb, None
    rep.ok(f"форма {emb.shape} — {len(q_ids)} query + {len(g_ids)} gallery")

    if not np.isfinite(emb).all():
        rep.fail(f"не-конечные значения: NaN {int(np.isnan(emb).sum())}, "
                 f"inf {int(np.isinf(emb).sum())}")
    else:
        rep.ok("NaN и inf отсутствуют")

    norms = np.linalg.norm(emb, axis=1)
    if np.allclose(norms, 1.0, atol=1e-3):
        rep.ok(f"векторы L2-нормированы (нормы {norms.min():.4f}..{norms.max():.4f})")
    else:
        rep.warn(f"нормы не единичные: {norms.min():.4f}..{norms.max():.4f} — "
                 f"поиск по косинусу подразумевает нормировку")
    if (norms < 1e-6).any():
        rep.fail(f"нулевых векторов: {int((norms < 1e-6).sum())}")

    uniq = len(np.unique(emb, axis=0))
    if uniq != len(emb):
        rep.warn(f"одинаковых векторов: {len(emb) - uniq} — возможны дубли кадров")
    else:
        rep.ok("одинаковых векторов нет")

    idx_path = art / "embeddings_index.csv"
    order_ok = False
    if not idx_path.exists():
        rep.fail("нет embeddings_index.csv — порядок строк нечем подтвердить")
    else:
        with idx_path.open() as f:
            saved = [r["image_id"] for r in csv.DictReader(f)]
        want = list(q_ids) + list(g_ids)
        if saved == want:
            rep.ok("порядок строк канонический: test_query.csv, затем test_gallery.csv")
            order_ok = True
        elif sorted(saved) == sorted(want):
            rep.fail("порядок строк НЕ канонический — организаторы сверяют "
                     "submission.csv именно по этому порядку")
        else:
            rep.fail("в индексе другой набор объектов")
    return emb, order_ok


def check_candidates(art: Path, q_ids, g_ids, rep: Report, threshold=None):
    section("candidates.csv")
    p = art / "candidates.csv"
    if not p.exists():
        rep.fail(f"нет файла {p}")
        return

    with p.open() as f:
        rdr = csv.reader(f)
        header = next(rdr, None)
        rows = list(rdr)
    if header != CAND_HEADER:
        rep.fail(f"заголовок {header}, ожидался {CAND_HEADER}")
    else:
        rep.ok("заголовок корректный")

    qset, gset = set(q_ids), set(g_ids)
    bad_q = {r[0] for r in rows if r[0] not in qset}
    bad_g = {r[1] for r in rows if r[1] not in gset}
    if bad_q:
        rep.fail(f"query_id вне test_query.csv: {len(bad_q)}")
    if bad_g:
        rep.fail(f"gallery_id вне test_gallery.csv: {len(bad_g)}")
    if not bad_q and not bad_g:
        rep.ok("все идентификаторы существуют")

    pairs = [(r[0], r[1]) for r in rows]
    if len(set(pairs)) != len(pairs):
        rep.fail(f"дублирующихся пар query-gallery: {len(pairs) - len(set(pairs))}")
    else:
        rep.ok("дублирующихся пар нет")

    try:
        conf = [float(r[2]) for r in rows]
    except (ValueError, IndexError):
        rep.fail("confidence не приводится к числу")
        return
    if min(conf) < -1.0001 or max(conf) > 1.0001:
        rep.fail(f"confidence вне [-1, 1]: {min(conf):.4f}..{max(conf):.4f}")
    else:
        rep.ok(f"confidence в диапазоне {min(conf):.4f}..{max(conf):.4f}")

    by_q = {}
    for r in rows:
        by_q.setdefault(r[0], []).append(float(r[2]))
    too_many = sum(1 for v in by_q.values() if len(v) > TOP_K)
    if too_many:
        rep.warn(f"query с более чем {TOP_K} кандидатами: {too_many}")
    unsorted = sum(1 for v in by_q.values() if any(a < b - 1e-9 for a, b in zip(v, v[1:])))
    if unsorted:
        rep.fail(f"query, где confidence не убывает: {unsorted}")
    else:
        rep.ok("внутри каждого query confidence убывает")

    refused = len(qset) - len(by_q)
    rep.ok(f"принято query {len(by_q)}, отказов {refused} "
           f"({refused / len(qset) * 100:.1f}%)")
    if threshold is not None:
        below = sum(1 for c in conf if c < threshold - 1e-9)
        if below:
            rep.fail(f"строк ниже порога {threshold}: {below}")
        else:
            rep.ok(f"все строки не ниже порога {threshold}")


def check_postprocess(art: Path, emb, sub, q_ids, g_ids, rep: Report):
    """Если выдача пересортирована — она обязана воспроизводиться из векторов.

    Организаторы по embeddings.npy проверяют, что submission.csv получен
    моделью. Постобработка меняет порядок, поэтому простое совпадение топ-1
    тут не работает. Вместо него проверяется более сильное свойство: взять
    embeddings.npy, применить записанный рядом конвейер с записанными
    параметрами и получить ту же таблицу. Это и есть доказательство
    происхождения, причём проверяемое кем угодно.
    """
    pp_path = art / "postprocess.json"
    if not pp_path.exists():
        return False
    section("постобработка выдачи")
    meta = json.loads(pp_path.read_text())
    rep.ok(f"конвейер: {meta.get('pipeline')}, параметры {meta.get('params')}")

    sys.path.insert(0, str(art.parent / "research"))
    try:
        import reid_postprocess as pp
    except ImportError as exc:
        rep.fail(f"нечем воспроизвести постобработку: {exc}")
        return True

    nq = len(q_ids)
    par = meta.get("params", {})
    sim, _, _ = pp.rank([emb[:nq]], [emb[nq:]], weights=[1.0],
                        qq_k=par.get("qq_k", 1), qq_alpha=par.get("qq_alpha", 3.0),
                        csls_k=par.get("csls_k", 5),
                        rerank_lambda=par.get("rerank_lambda", 0.7),
                        rerank_k1=par.get("rerank_k1", 30),
                        rerank_k2=par.get("rerank_k2", 6))
    order = np.argsort(-sim, axis=1)[:, :TOP_K]
    bad = sum(1 for i, qid in enumerate(q_ids)
              if [g_ids[j] for j in order[i]] != sub[qid])
    if bad == 0:
        rep.ok(f"submission.csv воспроизводится из embeddings.npy точь-в-точь "
               f"на всех {nq} запросах")
    else:
        rep.fail(f"submission.csv не воспроизводится из embeddings.npy: "
                 f"расходится {bad} строк из {nq}")
    return True


def check_consistency(emb, order_ok, sub, q_ids, g_ids, rep: Report, sample=200):
    """Сквозная проверка: та же, что сделают организаторы."""
    section("сквозная проверка submission.csv против embeddings.npy")
    if emb is None or sub is None:
        rep.warn("нечего сверять — один из файлов не прочитан")
        return
    if order_ok is not True:
        rep.warn("порядок строк не подтверждён, сверка будет неверной — пропускаю")
        return

    nq = len(q_ids)
    q_emb, g_emb = emb[:nq], emb[nq:]
    rng = np.random.default_rng(0)
    pick = rng.choice(nq, size=min(sample, nq), replace=False)
    sim = q_emb[pick] @ g_emb.T
    top1 = np.argmax(sim, axis=1)

    mismatch = 0
    for row, gi in zip(pick, top1):
        claimed = sub[q_ids[row]][0]
        if claimed != g_ids[gi]:
            mismatch += 1
    share = mismatch / len(pick)
    if mismatch == 0:
        rep.ok(f"топ-1 из векторов совпал с submission.csv на всех {len(pick)} пробах")
    elif share <= 0.02:
        rep.warn(f"топ-1 расходится на {mismatch} из {len(pick)} проб "
                 f"({share*100:.1f}%) — допустимо при пересортировке выдачи")
    else:
        rep.fail(f"топ-1 расходится на {mismatch} из {len(pick)} проб "
                 f"({share*100:.1f}%) — submission.csv не выводится из embeddings.npy")


def check_git(art: Path, rep: Report):
    section("артефакты в репозитории")
    names = ["submission.csv", "embeddings.npy", "candidates.csv", "embeddings_index.csv"]
    try:
        tracked = subprocess.run(
            ["git", "ls-files", "--"] + [str(art / n) for n in names],
            capture_output=True, text=True, cwd=art.parent, timeout=20,
        ).stdout.split()
    except Exception as exc:
        rep.warn(f"git недоступен ({exc}) — проверку пропускаю")
        return
    tracked_names = {Path(t).name for t in tracked}
    for n in names:
        if n in tracked_names:
            rep.ok(f"{n} под контролем версий")
        else:
            rep.fail(f"{n} НЕ в репозитории — жюри его не увидит")


def main() -> int:
    ap = argparse.ArgumentParser()
    here = Path(__file__).resolve().parent.parent
    ap.add_argument("--artifacts", default=str(here / "artifacts"))
    ap.add_argument("--data", default="/opt/case17/data",
                    help="каталог с test_query.csv и test_gallery.csv")
    ap.add_argument("--threshold", type=float, default=None,
                    help="порог отказа, если нужно проверить candidates.csv на него")
    ap.add_argument("--strict", action="store_true", help="предупреждения тоже роняют")
    a = ap.parse_args()

    art, data = Path(a.artifacts), Path(a.data)
    rep = Report()

    section("исходные файлы датасета")
    q_csv, g_csv = data / "test_query.csv", data / "test_gallery.csv"
    for p in (q_csv, g_csv):
        if not p.exists():
            print(f"  \033[31mFAIL\033[0m нет {p} — без него порядок не проверить",
                  file=sys.stderr)
            return 2
    q_ids, g_ids = read_ids(q_csv), read_ids(g_csv)
    rep.ok(f"test_query.csv {len(q_ids)}, test_gallery.csv {len(g_ids)}")
    if len(set(q_ids)) != len(q_ids) or len(set(g_ids)) != len(g_ids):
        rep.fail("в исходных csv есть дубли image_id")

    sub = check_submission(art, q_ids, g_ids, rep)
    emb, order_ok = check_embeddings(art, q_ids, g_ids, rep)
    check_candidates(art, q_ids, g_ids, rep, a.threshold)
    # Постобработка и прямой топ-1 — взаимоисключающие способы доказать
    # происхождение submission.csv. Что применимо, то и проверяем.
    if not check_postprocess(art, emb, sub, q_ids, g_ids, rep):
        check_consistency(emb, order_ok, sub, q_ids, g_ids, rep)
    check_git(art, rep)

    section("ИТОГ")
    print(f"  успешно {len(rep.oks)}, предупреждений {len(rep.warns)}, "
          f"ошибок {len(rep.fails)}")
    if rep.fails:
        print("\033[31m  АРТЕФАКТЫ НЕВАЛИДНЫ — коммитить нельзя\033[0m")
        for m in rep.fails:
            print(f"    - {m}")
        return 1
    if rep.warns and a.strict:
        print("\033[33m  есть предупреждения, а запрошен --strict\033[0m")
        return 1
    print("\033[32m  АРТЕФАКТЫ ВАЛИДНЫ\033[0m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
