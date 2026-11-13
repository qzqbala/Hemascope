"""Ручные морфологические признаки лейкоцита — baseline для классификатора типов.

Признаки повторяют то, на что смотрит лаборант: размер и форма ядра, число долек,
цвет и зернистость цитоплазмы.
"""
from __future__ import annotations

import cv2
import numpy as np

FEATURE_NAMES = [
    "nucleus_area", "n_lobes", "nucleus_solidity", "nucleus_eccentricity", "nucleus_perimeter_ratio",
    "nuc_h", "nuc_s", "nuc_v", "cyto_h", "cyto_s", "cyto_v", "cyto_red_ratio",
    "texture_std", "texture_lap", "nc_ratio",
] + [f"hue_hist_{i}" for i in range(8)]


def features(crop_rgb: np.ndarray) -> np.ndarray:
    img = cv2.resize(crop_rgb, (128, 128), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)
    r, g, b = [img[..., i].astype(np.float32) for i in range(3)]
    stain = np.clip(b - g, 0, 255).astype(np.uint8)
    thr, nuc = cv2.threshold(cv2.GaussianBlur(stain, (5, 5), 0), 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    nuc = cv2.morphologyEx(nuc, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    # клетка целиком: всё, что темнее фона
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    _, cell = cv2.threshold(cv2.GaussianBlur(gray, (7, 7), 0), 0, 1, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    cell = np.maximum(cell, nuc)
    cyto = (cell > 0) & (nuc == 0)

    n_comp, lab, stats, _ = cv2.connectedComponentsWithStats(nuc)
    big = [i for i in range(1, n_comp) if stats[i, cv2.CC_STAT_AREA] > 30]
    area = nuc.sum() / nuc.size
    cnts, _ = cv2.findContours(nuc, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if cnts:
        allc = np.vstack(cnts)
        hull = cv2.convexHull(allc)
        solidity = nuc.sum() / max(cv2.contourArea(hull), 1)
        perim = sum(cv2.arcLength(c, True) for c in cnts)
        perim_ratio = perim / (2 * np.sqrt(np.pi * max(nuc.sum(), 1)))
        if len(allc) >= 5:
            (_, _), (ma, mi), _ = cv2.fitEllipse(allc)
            ecc = np.sqrt(1 - (min(ma, mi) / max(ma, mi, 1e-6)) ** 2)
        else:
            ecc = 0.0
    else:
        solidity, perim_ratio, ecc = 0.0, 0.0, 0.0

    def mean_in(mask, ch):
        return float(ch[mask].mean()) if mask.any() else 0.0

    nucm = nuc > 0
    hue_hist = np.histogram(hsv[..., 0][cell > 0], bins=8, range=(0, 180))[0]
    hue_hist = hue_hist / max(hue_hist.sum(), 1)
    return np.array([
        area, len(big), solidity, ecc, perim_ratio,
        mean_in(nucm, hsv[..., 0]), mean_in(nucm, hsv[..., 1]), mean_in(nucm, hsv[..., 2]),
        mean_in(cyto, hsv[..., 0]), mean_in(cyto, hsv[..., 1]), mean_in(cyto, hsv[..., 2]),
        mean_in(cyto, r) / (mean_in(cyto, b) + 1),
        float(gray[cell > 0].std()) if (cell > 0).any() else 0.0,
        float(cv2.Laplacian(gray, cv2.CV_32F)[cell > 0].var()) if (cell > 0).any() else 0.0,
        area / max(cell.mean(), 1e-6),
        *hue_hist,
    ], dtype=np.float32)
