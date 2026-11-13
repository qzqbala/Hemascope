"""Устойчивость к плохим снимкам: как падает качество детектора при размытии / затемнении
и успевает ли проверка качества отклонить снимок раньше, чем модель начнёт ошибаться.
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from calibrate_quality import degrade  # noqa: E402
from hemascope.data import CLASS_NAMES, load_split  # noqa: E402
from hemascope.metrics import detection_prf  # noqa: E402
from hemascope.env import device  # noqa: E402
from hemascope.pipeline import check_quality  # noqa: E402

DEVICE = device()
from hemascope.plotting import COLORS, style  # noqa: E402

NAME = "yolo11n_pretrained"
LEVELS = {"blur": [0, 1, 2, 3, 4, 6], "dark": [1.0, 0.8, 0.6, 0.45, 0.3]}


def main():
    from ultralytics import YOLO

    style()
    out = ROOT / "reports" / "robustness"
    out.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(ROOT / "runs" / "detect" / NAME / "weights" / "best.pt"))
    met = json.loads((ROOT / "reports" / "detector" / NAME / "metrics.json").read_text())
    thr, imgsz = met["thresholds_from_val"], met.get("imgsz", 416)
    limits = json.loads((ROOT / "reports" / "quality" / "quality_limits.json").read_text())
    samples = load_split("test")
    imgs = [np.asarray(Image.open(s.image).convert("RGB")) for s in samples]
    if "--plot-only" in sys.argv:
        plot(json.loads((out / "robustness.json").read_text()), out)
        return
    res = {}
    for kind, levels in LEVELS.items():
        res[kind] = []
        for lvl in levels:
            recs, rej = [], 0
            for s, im in zip(samples, imgs):
                x = degrade(im, kind, lvl) if (kind == "blur" and lvl > 0) or (kind == "dark" and lvl < 1) else im
                rej += not check_quality(x, **limits).ok
                r = model.predict(x[..., ::-1], imgsz=imgsz, conf=min(thr.values()), iou=0.6, device=DEVICE, verbose=False)[0]
                c, sc = r.boxes.cls.cpu().numpy().astype(int), r.boxes.conf.cpu().numpy()
                keep = np.array([v >= thr[CLASS_NAMES[k]] for k, v in zip(c, sc)], dtype=bool)
                recs.append({"gt_boxes": s.boxes, "gt_cls": s.classes, "boxes": r.boxes.xyxy.cpu().numpy()[keep],
                             "cls": c[keep], "scores": sc[keep]})
            d = detection_prf(recs)
            res[kind].append({"level": lvl, "rejected": rej / len(samples),
                              **{n: d[n]["f1"] for n in CLASS_NAMES}})
            print(kind, lvl, res[kind][-1])
    (out / "robustness.json").write_text(json.dumps(res, ensure_ascii=False, indent=2))
    plot(res, out)


def plot(res, out):
    style()
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for ax, kind, xlabel in [(axes[0], "blur", "Размытие, σ (пикселей при стороне 400)"),
                             (axes[1], "dark", "Яркость, доля исходной")]:
        lv = [r["level"] for r in res[kind]]
        for c, n in enumerate(CLASS_NAMES):
            ax.plot(lv, [r[n] for r in res[kind]], marker="o", color=COLORS[c],
                    label={"WBC": "F1 лейкоциты", "RBC": "F1 эритроциты", "Platelets": "F1 тромбоциты"}[n])
        ax.bar(lv, [r["rejected"] for r in res[kind]], width=(max(lv) - min(lv)) / 12, color="#CBD5E0",
               label="доля отклонённых проверкой качества", zorder=0)
        ax.set_xlabel(xlabel)
        ax.set_ylim(0, 1.02)
        if kind == "dark":
            ax.invert_xaxis()
    axes[0].set_title("Размытие: F1 падает медленно, проверка отсекает сильное", loc="left")
    axes[1].set_title("Затемнение: проверка срабатывает раньше, чем падает F1", loc="left")
    axes[0].legend(frameon=False, fontsize=8, loc="lower left")
    fig.savefig(out / "robustness.png")


if __name__ == "__main__":
    main()
