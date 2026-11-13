"""Классический CV-baseline без нейросетей.

Идея (как у ранних систем автоматического анализа мазка):
1. Масштабируем снимок так, чтобы короткая сторона была 400 px — размеры клеток относительно
   снимка в датасете стабильны (эритроцит ≈ 0,21, лейкоцит ≈ 0,40, тромбоцит ≈ 0,085 короткой стороны).
2. Ядра лейкоцитов и тромбоциты окрашены в фиолетовый: признак «синева минус зелень» (B − G),
   порог Оцу → связные компоненты. Крупные → лейкоцит, мелкие → тромбоцит.
3. Эритроциты — круги преобразования Хафа с радиусом из известного диапазона.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

S = 400  # короткая сторона после масштабирования


@dataclass
class BaselineParams:
    hough_param2: float = 18.0   # порог аккумулятора Хафа: меньше → больше кругов
    rbc_rmin: float = 0.075      # радиусы эритроцитов, доля короткой стороны
    rbc_rmax: float = 0.13
    rbc_mindist: float = 0.15
    stain_offset: float = 0.0    # сдвиг порога окраски относительно Оцу
    wbc_min_area: float = 0.012  # площадь компоненты / S² для лейкоцита
    plt_min_area: float = 0.0012
    plt_max_area: float = 0.008


def detect(image_rgb: np.ndarray, p: BaselineParams = BaselineParams()):
    """Возвращает (boxes [N,4] в координатах исходного снимка, classes [N], scores [N])."""
    h0, w0 = image_rgb.shape[:2]
    k = S / min(h0, w0)
    img = cv2.resize(image_rgb, (round(w0 * k), round(h0 * k)), interpolation=cv2.INTER_AREA)
    h, w = img.shape[:2]
    r, g, b = [img[..., i].astype(np.float32) for i in range(3)]

    # --- окрашенные структуры (ядра лейкоцитов, тромбоциты)
    stain = cv2.GaussianBlur(np.clip(b - g, 0, 255).astype(np.uint8), (5, 5), 0)
    thr, _ = cv2.threshold(stain, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thr = max(thr + p.stain_offset, 25)
    mask = (stain > thr).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    n, lab, stats, cent = cv2.connectedComponentsWithStats(mask)
    boxes, classes, scores = [], [], []
    wbc_centers = []
    # от крупных компонент к мелким; дольки сегментированного ядра внутри уже найденного
    # лейкоцита не считаем отдельными клетками
    for i in sorted(range(1, n), key=lambda j: -stats[j, cv2.CC_STAT_AREA]):
        area = stats[i, cv2.CC_STAT_AREA] / (S * S)
        x, y, bw, bh = stats[i, :4]
        cx, cy = cent[i]
        if any(abs(cx - wx) < wh and abs(cy - wy) < wh for wx, wy, wh in wbc_centers):
            continue
        if area >= p.wbc_min_area:
            half = max(bw, bh, 0.40 * S) / 2  # ядро → весь лейкоцит с цитоплазмой
            boxes.append([cx - half, cy - half, cx + half, cy + half]); classes.append(0)
            scores.append(min(1.0, area / 0.05)); wbc_centers.append((cx, cy, half))
        elif p.plt_min_area <= area <= p.plt_max_area:
            half = max(bw, bh, 0.085 * S) / 2
            boxes.append([cx - half, cy - half, cx + half, cy + half]); classes.append(2)
            scores.append(0.5)

    # --- эритроциты: круги Хафа
    gray = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), (7, 7), 1.5)
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=p.rbc_mindist * S,
                               param1=60, param2=p.hough_param2,
                               minRadius=int(p.rbc_rmin * S), maxRadius=int(p.rbc_rmax * S))
    if circles is not None:
        for cx, cy, rad in circles[0]:
            # круг, чей центр внутри лейкоцита, — это сам лейкоцит
            if any((cx - wx) ** 2 + (cy - wy) ** 2 < (0.8 * wh) ** 2 for wx, wy, wh in wbc_centers):
                continue
            boxes.append([cx - rad, cy - rad, cx + rad, cy + rad]); classes.append(1); scores.append(0.5)

    boxes = np.array(boxes, dtype=np.float32).reshape(-1, 4)
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, w) / k
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, h) / k
    return boxes, np.array(classes, dtype=int), np.array(scores, dtype=np.float32)
