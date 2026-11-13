"""Собственный расчёт AP / mAP (COCO-стиль, 101 точка) — нужен для bootstrap-интервалов по снимкам.

Сверяется со штатным валидатором Ultralytics в scripts/deep_analysis.py.
"""
from __future__ import annotations

import numpy as np

from .data import CLASS_NAMES
from .metrics import iou_matrix

IOU_THRS = np.round(np.arange(0.5, 0.96, 0.05), 2)


def _image_matches(pred_boxes, pred_scores, gt_boxes, thr):
    """Для каждого предсказания (по убыванию уверенности): TP при пороге IoU thr."""
    order = np.argsort(-pred_scores)
    ious = iou_matrix(pred_boxes[order], gt_boxes)
    used = np.zeros(len(gt_boxes), bool)
    tp = np.zeros(len(order), bool)
    for i in range(len(order)):
        if ious.shape[1] == 0:
            break
        cand = np.where(~used & (ious[i] >= thr))[0]
        if len(cand):
            j = cand[np.argmax(ious[i, cand])]
            used[j] = True
            tp[i] = True
    return pred_scores[order], tp


def precompute(records: list[dict]):
    """Для каждого снимка, класса и порога IoU: (scores, tp, n_gt) — чтобы bootstrap не пересчитывал IoU."""
    cache = []
    for r in records:
        per = {}
        for c in range(len(CLASS_NAMES)):
            pm, gm = r["cls"] == c, r["gt_cls"] == c
            pb, ps, gb = r["boxes"][pm], r["scores"][pm], r["gt_boxes"][gm]
            per[c] = {"n_gt": int(gm.sum()),
                      "by_thr": {t: _image_matches(pb, ps, gb, t) for t in IOU_THRS}}
        cache.append(per)
    return cache


def _ap(scores, tp, n_gt):
    if n_gt == 0:
        return np.nan
    if len(scores) == 0:
        return 0.0
    o = np.argsort(-scores, kind="stable")
    tp = tp[o]
    ctp, cfp = np.cumsum(tp), np.cumsum(~tp)
    rec, prec = ctp / n_gt, ctp / np.maximum(ctp + cfp, 1e-9)
    prec = np.maximum.accumulate(prec[::-1])[::-1]
    rs = np.linspace(0, 1, 101)
    idx = np.searchsorted(rec, rs, side="left")
    return float(np.mean([prec[i] if i < len(prec) else 0.0 for i in idx]))


def map_from_cache(cache, idx=None) -> dict:
    idx = range(len(cache)) if idx is None else idx
    sel = [cache[i] for i in idx]
    out = {}
    for c, name in enumerate(CLASS_NAMES):
        n_gt = sum(s[c]["n_gt"] for s in sel)
        aps = []
        for t in IOU_THRS:
            sc = np.concatenate([s[c]["by_thr"][t][0] for s in sel]) if sel else np.array([])
            tp = np.concatenate([s[c]["by_thr"][t][1] for s in sel]) if sel else np.array([], bool)
            aps.append(_ap(sc, tp, n_gt))
        out[name] = {"AP50": aps[0], "AP50_95": float(np.nanmean(aps)) if not np.all(np.isnan(aps)) else np.nan}
    out["mAP50"] = float(np.nanmean([out[n]["AP50"] for n in CLASS_NAMES]))
    out["mAP50_95"] = float(np.nanmean([out[n]["AP50_95"] for n in CLASS_NAMES]))
    return out


def bootstrap_map(cache, n_boot=1000, seed=0):
    rng = np.random.default_rng(seed)
    vals = {"mAP50": [], "mAP50_95": []}
    for _ in range(n_boot):
        m = map_from_cache(cache, rng.integers(0, len(cache), len(cache)))
        vals["mAP50"].append(m["mAP50"])
        vals["mAP50_95"].append(m["mAP50_95"])
    return {k: [float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975))] for k, v in vals.items()}
