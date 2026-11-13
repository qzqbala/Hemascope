"""Калибровка проверки качества снимка и её проверка на искусственно испорченных снимках.

Пороги = 1-й перцентиль резкости (×0,9) и 0,5/99,5-й перцентили яркости (±10) по train.
Затем на val проверяем: какую долю размытых / затемнённых / пересвеченных снимков система отклоняет
и какую долю нормальных пропускает.
"""
import json
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from hemascope.data import load_split  # noqa: E402
from hemascope.pipeline import check_quality  # noqa: E402
from hemascope.plotting import COLORS, style  # noqa: E402

OUT = ROOT / "reports" / "quality"


def degrade(img, kind, level):
    if kind == "blur":
        return cv2.GaussianBlur(img, (0, 0), level * min(img.shape[:2]) / 400)
    if kind == "dark":
        return np.clip(img * level, 0, 255).astype(np.uint8)
    if kind == "bright":
        return np.clip(img + (255 - img) * level, 0, 255).astype(np.uint8)
    return img


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    style()
    tr = [np.asarray(Image.open(s.image).convert("RGB")) for s in load_split("train")]
    q = [check_quality(x, 0, 0, 256) for x in tr]
    sharp = np.array([x.sharpness for x in q])
    bright = np.array([x.brightness for x in q])
    limits = {"min_sharp": float(np.percentile(sharp, 1) * 0.9),
              "min_bright": float(np.percentile(bright, 0.5) - 10),
              "max_bright": float(min(np.percentile(bright, 99.5) + 10, 250))}
    (OUT / "quality_limits.json").write_text(json.dumps(limits, indent=2))

    val = [np.asarray(Image.open(s.image).convert("RGB")) for s in load_split("val")]
    tests = {"clean": [("none", 0)], "blur": [("blur", s) for s in (1, 2, 3, 4)],
             "dark": [("dark", f) for f in (0.6, 0.45, 0.3)], "bright": [("bright", f) for f in (0.5, 0.7, 0.85)]}
    res = {"limits": limits, "train_sharpness_pct": {p: float(np.percentile(sharp, p)) for p in (1, 5, 50)},
           "rejection_rate": {}}
    for name, cfgs in tests.items():
        for kind, lvl in cfgs:
            rej = np.mean([not check_quality(degrade(x, kind, lvl), **limits).ok for x in val])
            res["rejection_rate"][f"{kind}_{lvl}"] = float(rej)
    (OUT / "quality_check.json").write_text(json.dumps(res, ensure_ascii=False, indent=2))

    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.hist(sharp, bins=50, color=COLORS[2], alpha=0.8, label="обучающие снимки")
    ax.axvline(limits["min_sharp"], color=COLORS[1], lw=2, label="порог «переснимите»")
    ax.set_xlabel("Резкость: |снимок − размытая копия| / контраст")
    ax.set_ylabel("Снимков")
    ax.set_title("Порог резкости ниже всех обучающих снимков", loc="left")
    ax.legend(frameon=False)
    fig.savefig(OUT / "quality_calibration.png")
    print(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
