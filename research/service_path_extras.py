#!/usr/bin/env python3
"""Признаки бэкбона считаются один раз, дальше всё дёшево.

Зачем отдельный скрипт: найден перекос обучения и инференса. Голова училась
на L2-нормированных выходах бэкбона (extract_features.py нормирует перед
сохранением), а сервис подаёт в голову сырой выход. Это и давало косинус 0.57
между выгрузкой признаков и ответом сервиса.

Кэшируется выход БЭКБОНА (до головы), поэтому обе версии головы считаются
за секунды, а не за одиннадцать минут.

  * ранжирование: сырой вход в голову против L2-нормированного;
  * катастрофы до и после постобработки;
  * распределения максимального сходства для графика порога отказа.

    python research/service_path_extras.py
"""
import csv, json, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

HERE = Path('/opt/case17/app/research')
CROPS = Path('/opt/case17/data/crops/train')
sys.path.insert(0, str(HERE))
import eval_service_path as E
import reid_postprocess as PP

CACHE = HERE / 'results' / 'backbone_feats.npz'


@torch.inference_mode()
def backbone_feats(rows, model, tf, batch=16, label=''):
    """Выход бэкбона без головы, оригинал и отражение отдельно."""
    d = model.num_features
    o = np.zeros((len(rows), d), dtype='float32')
    f = np.zeros((len(rows), d), dtype='float32')
    t0 = time.perf_counter()
    for i in range(0, len(rows), batch):
        ch = rows[i:i + batch]
        ims = [Image.open(CROPS / f"{r['image_id']}.jpg").convert('RGB') for r in ch]
        x = torch.stack([tf(im) for im in ims])
        o[i:i + len(ch)] = model(x).numpy()
        f[i:i + len(ch)] = model(torch.flip(x, dims=[3])).numpy()
        if i and i % (batch * 10) == 0:
            print(f'    {label} {i}/{len(rows)}  {i/(time.perf_counter()-t0):.2f} кадр/с', flush=True)
    return o, f


def load(qry, gal):
    if CACHE.exists():
        z = np.load(CACHE)
        if len(z['qo']) == len(qry) and len(z['go']) == len(gal):
            print(f'выход бэкбона из кэша {CACHE.name}')
            return {k: z[k] for k in ('qo', 'qf', 'go', 'gf')}
    ck = torch.load('/opt/case17/models/reid_current.pt', map_location='cpu', weights_only=False)
    import backbones
    spec = backbones.get(ck['args'].get('backbone', 'siglip_base'))
    _, tf, model = backbones.build_encoder(spec)
    t0 = time.perf_counter()
    go, gf = backbone_feats(gal, model, tf, label='gallery')
    qo, qf = backbone_feats(qry, model, tf, label='query')
    print(f'бэкбон посчитан за {time.perf_counter()-t0:.0f} с', flush=True)
    d = dict(qo=qo, qf=qf, go=go, gf=gf)
    np.savez_compressed(CACHE, **d)
    return d


def head_out(z, head, prenorm):
    """Прогон через голову. prenorm=True — так, как её учили."""
    out = {}
    with torch.inference_mode():
        for k in ('q', 'g'):
            acc = None
            for suf in ('o', 'f'):
                x = torch.from_numpy(z[k + suf])
                if prenorm:
                    x = F.normalize(x, p=2, dim=1)
                v = head(x)[1]
                acc = v if acc is None else acc + v
            out[k] = F.normalize(acc, p=2, dim=1).numpy()
    return out['q'], out['g']


def scores(q, g, qry, gal, cross, seed=42):
    rng = np.random.default_rng(seed)
    vids = sorted({r['vehicle_id'] for r in qry})
    unknown = set(rng.choice(vids, size=len(vids)//2, replace=False).tolist())
    g_vid = np.array([r['vehicle_id'] for r in gal])
    g_cam = np.array([r['camera_id'] for r in gal])
    sim = q @ g.T
    pos, neg = [], []
    for i, qr in enumerate(qry):
        v = qr['vehicle_id']
        keep = (g_cam != qr['camera_id']) if cross else np.ones(len(g_vid), bool)
        if v in unknown:
            keep = keep & (g_vid != v)
        if not keep.any():
            continue
        (neg if v in unknown else pos).append(float(sim[i][keep].max()))
    return np.array(pos), np.array(neg)


def main():
    vs = list(csv.DictReader((HERE / 'val_split.csv').open()))
    gal = [r for r in vs if r['role'] == 'gallery']
    qry = [r for r in vs if r['role'] == 'query']
    print(f'галерея {len(gal)}, запросов {len(qry)}', flush=True)

    z = load(qry, gal)
    ck = torch.load('/opt/case17/models/reid_current.pt', map_location='cpu', weights_only=False)
    from head_model import ReidHead
    ar = ck['args']
    head = ReidHead(ar['in_dim'], ar['hidden'], ar['out_dim'], ar['dropout'])
    head.load_state_dict(ck['head']); head.eval()

    out = {}
    best = None
    for tag, prenorm in (('как сейчас в сервисе (сырой вход)', False),
                         ('как училась голова (L2 перед головой)', True)):
        q, g = head_out(z, head, prenorm)
        m = E.score(q @ g.T, qry, gal)
        key = 'prenorm' if prenorm else 'raw'
        out[key] = m
        print(f"{tag}: mAP {m['mAP']}  Rank-1 {m['rank1']}  Rank-5 {m['rank5']}  "
              f"катастроф {m['catastrophic_pct']}%", flush=True)
        if prenorm:
            best = (q, g)

    q, g = best
    t0 = time.perf_counter()
    sim_pp, _, _ = PP.rank([q], [g])
    post = E.score(sim_pp, qry, gal)
    out['prenorm_postprocess'] = post
    print(f"постобработка поверх: mAP {post['mAP']}  Rank-1 {post['rank1']}  "
          f"катастроф {post['catastrophic_pct']}%  ({time.perf_counter()-t0:.0f} с)", flush=True)

    npz = {}
    for name, cross in (('jury_cross_camera', True), ('deployment', False)):
        pos, neg = scores(q, g, qry, gal, cross)
        npz[f'{name}_pos'] = pos
        npz[f'{name}_neg'] = neg
        r = E.refusal(q, g, qry, gal, cross)
        out[f'refusal_{name}'] = r
        print(f"отказ «{name}»: AUC {r['auc']}  порог сбал. {r['balanced']['threshold']} "
              f"(F1 {r['balanced']['f1']}, TNR {r['balanced']['TNR']}), "
              f"строгий {r['strict']['threshold']} (TNR {r['strict']['TNR']})", flush=True)
    np.savez(HERE / 'results' / 'refusal_scores_service.npz', **npz)

    p = HERE / 'results' / 'service_path_extras.json'
    p.write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print(f'записано: {p}')


if __name__ == '__main__':
    main()
