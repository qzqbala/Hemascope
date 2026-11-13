"""Метрики: сопоставление боксов по IoU, P/R/F1, ошибки подсчёта, bootstrap-интервалы."""
from __future__ import annotations

import numpy as np

from .data import CLASS_NAMES


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)


def match(pred_boxes, pred_scores, gt_boxes, iou_thr=0.5):
    """Жадное сопоставление по убыванию уверенности. Возвращает (tp, fp, fn, matched_gt_mask, pred_is_tp)."""
    order = np.argsort(-np.asarray(pred_scores)) if len(pred_boxes) else np.array([], dtype=int)
    ious = iou_matrix(np.asarray(pred_boxes)[order] if len(order) else np.zeros((0, 4)), np.asarray(gt_boxes))
    used = np.zeros(len(gt_boxes), dtype=bool)
    is_tp = np.zeros(len(order), dtype=bool)
    for i in range(len(order)):
        if ious.shape[1] == 0:
            break
        cand = np.where(~used & (ious[i] >= iou_thr))[0]
        if len(cand):
            j = cand[np.argmax(ious[i, cand])]
            used[j] = True
            is_tp[i] = True
    pred_is_tp = np.zeros(len(order), dtype=bool)
    pred_is_tp[order] = is_tp
    tp = int(is_tp.sum())
    return tp, len(order) - tp, int((~used).sum()), used, pred_is_tp


def detection_prf(records: list[dict], iou_thr=0.5) -> dict:
    """records: [{gt_boxes, gt_cls, boxes, cls, scores}] → P/R/F1 по классам при IoU ≥ iou_thr."""
    out = {}
    for c, name in enumerate(CLASS_NAMES):
        tp = fp = fn = 0
        for r in records:
            pm, gm = r["cls"] == c, r["gt_cls"] == c
            a, b, d, _, _ = match(r["boxes"][pm], r["scores"][pm], r["gt_boxes"][gm], iou_thr)
            tp, fp, fn = tp + a, fp + b, fn + d
        p = tp / (tp + fp) if tp + fp else 0.0
        rcl = tp / (tp + fn) if tp + fn else 0.0
        f1 = (2 * p * rcl / (p + rcl) if p + rcl else 0.0) if tp + fp + fn else None  # None: клеток нет и не найдено
        out[name] = {"precision": p, "recall": rcl, "f1": f1,
                     "tp": tp, "fp": fp, "fn": fn}
    return out


def count_errors(true_counts: np.ndarray, pred_counts: np.ndarray) -> dict:
    """Ошибки подсчёта на снимок: MAE, относительная ошибка суммарного подсчёта, доля снимков с точным числом."""
    t, p = np.asarray(true_counts, float), np.asarray(pred_counts, float)
    diff = p - t
    return {
        "mae": float(np.abs(diff).mean()),
        "bias": float(diff.mean()),
        "exact": float((diff == 0).mean()),
        "total_rel_error": float((p.sum() - t.sum()) / max(t.sum(), 1)),
        "mape_nonzero": float(np.mean(np.abs(diff[t > 0]) / t[t > 0])) if (t > 0).any() else float("nan"),
    }


def bootstrap_ci(values_fn, n_items: int, n_boot=1000, seed=0, alpha=0.05):
    """values_fn(idx) → число; бутстреп по снимкам."""
    rng = np.random.default_rng(seed)
    stats = [values_fn(rng.integers(0, n_items, n_items)) for _ in range(n_boot)]
    return float(np.quantile(stats, alpha / 2)), float(np.quantile(stats, 1 - alpha / 2))
