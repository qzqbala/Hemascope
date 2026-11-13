"""Постобработка по протоколу разметки TXL-PBC.

Протокол (data/txl-pbc/manual_annotation_protocol.pdf): «For edge cells, annotate only if more than
half is visible». Детектор же находит и клетки, от которых в кадре видна лишь узкая полоска, —
по разметке это ложные срабатывания, а для подсчёта двойной учёт с соседним полем зрения.

Правило: бокс, касающийся края снимка, оставляем, только если его площадь ≥ ratio × типичной
площади целой клетки того же класса на этом снимке (медиана боксов, не касающихся края).
"""
from __future__ import annotations

import numpy as np

# запасной размер целой клетки (доля короткой стороны), если на снимке нет целых клеток класса
PRIOR_SIDE = {0: 0.40, 1: 0.21, 2: 0.085}


def touches_edge(boxes: np.ndarray, w: int, h: int, tol: float = 2.0) -> np.ndarray:
    return (boxes[:, 0] <= tol) | (boxes[:, 1] <= tol) | (boxes[:, 2] >= w - tol) | (boxes[:, 3] >= h - tol)


def edge_filter(boxes, classes, scores, w, h, ratio=0.5, classes_to_filter=(1,)):
    if len(boxes) == 0:
        return boxes, classes, scores
    keep = np.ones(len(boxes), bool)
    edge = touches_edge(boxes, w, h)
    area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    for c in classes_to_filter:
        m = classes == c
        if not (m & edge).any():
            continue
        full = m & ~edge
        ref = np.median(area[full]) if full.sum() >= 3 else (PRIOR_SIDE[c] * min(w, h)) ** 2
        keep &= ~(m & edge & (area < ratio * ref))
    return boxes[keep], classes[keep], scores[keep]
