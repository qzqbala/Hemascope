"""Классический baseline: подбор параметров на train, оценка на val и test.

Параметры подбираются по сетке на 200 случайных снимках train (средний F1 по трём классам),
затем один раз проверяются на val и test — так же, как для нейросети.
"""
import itertools
import time
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from hemascope.baseline import BaselineParams, detect  # noqa: E402
from hemascope.data import CLASS_NAMES, load_split  # noqa: E402
from hemascope.evaluate import records_to_report, save_predictions  # noqa: E402
from hemascope.metrics import detection_prf  # noqa: E402

OUT = ROOT / "reports" / "baseline"


TIMES = []


def run(samples, params):
    recs = []
    for s in samples:
        img = np.asarray(Image.open(s.image).convert("RGB"))
        t0 = time.perf_counter()
        b, c, sc = detect(img, params)
        TIMES.append(time.perf_counter() - t0)
        recs.append({"image": s.image.name, "gt_boxes": s.boxes, "gt_cls": s.classes,
                     "boxes": b, "cls": c, "scores": sc})
    return recs


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    train = load_split("train")
    tune = [train[i] for i in rng.choice(len(train), 200, replace=False)]
    grid = {"hough_param2": [10, 12, 14, 18, 24], "stain_offset": [15, 30, 45, 60],
            "rbc_rmin": [0.07, 0.085], "rbc_rmax": [0.12, 0.14]}
    best, best_f1, log = None, -1, []
    for vals in itertools.product(*grid.values()):
        p = BaselineParams(**dict(zip(grid.keys(), vals)))
        prf = detection_prf(run(tune, p))
        f1 = float(np.mean([prf[n]["f1"] for n in CLASS_NAMES]))
        log.append({**dict(zip(grid.keys(), vals)), "mean_f1": f1})
        if f1 > best_f1:
            best, best_f1 = p, f1
    print("best params", best, round(best_f1, 3))
    result = {"params": best.__dict__, "tune_mean_f1": best_f1, "grid": log}
    # то же правило краевых клеток, что и для нейросети: доля подбирается на val
    from evaluate_detector import EDGE_RATIOS, edge_rule
    val = run(load_split("val"), best)
    curve = {r: detection_prf(edge_rule(val, "val", r))["RBC"]["f1"] for r in EDGE_RATIOS}
    ratio = max(curve, key=curve.get)
    result["edge_rule"] = {"ratio_from_val": ratio, "val_rbc_f1_by_ratio": curve,
                           "val_rbc_f1_without_rule": detection_prf(val)["RBC"]["f1"]}
    result["val"] = records_to_report(edge_rule(val, "val", ratio), with_ci=False)
    TIMES.clear()
    test_plain = run(load_split("test"), best)
    result["latency_ms_median_cpu"] = float(np.median(TIMES) * 1000)
    test = edge_rule(test_plain, "test", ratio)
    result["test_without_edge_rule"] = records_to_report(test_plain, with_ci=False)
    result["test"] = records_to_report(test)
    save_predictions(test, OUT / "preds_test.json")
    (OUT / "baseline_metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps({k: result[k] for k in ("val", "test")}, ensure_ascii=False, indent=1)[:3000])


if __name__ == "__main__":
    main()
