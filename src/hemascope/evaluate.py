"""Единая оценка любого детектора (YOLO или классического) на одинаковых метриках."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .data import CLASS_NAMES, source_map
from .metrics import bootstrap_ci, count_errors, detection_prf


def records_to_report(records: list[dict], with_ci: bool = True) -> dict:
    """P/R/F1 при IoU 0,5 + ошибки подсчёта по классам + разбивка по источникам."""
    src = source_map().set_index("image")["source"]
    rep = {"n_images": len(records), "detection": detection_prf(records), "counting": {}}
    for c, name in enumerate(CLASS_NAMES):
        t = np.array([(r["gt_cls"] == c).sum() for r in records])
        p = np.array([(r["cls"] == c).sum() for r in records])
        ce = count_errors(t, p)
        if with_ci:
            ce["mae_ci95"] = bootstrap_ci(lambda idx: float(np.abs(p[idx] - t[idx]).mean()), len(t))
        rep["counting"][name] = ce
    rep["macro_f1"] = float(np.mean([rep["detection"][n]["f1"] for n in CLASS_NAMES]))
    by_src = {}
    for s in sorted(set(src.values)):
        sub = [r for r in records if src.get(r["image"]) == s]
        if sub:
            d = detection_prf(sub)
            by_src[s] = {"n_images": len(sub),
                         **{n: (round(d[n]["f1"], 4) if d[n]["f1"] is not None else None) for n in CLASS_NAMES},
                         "gt_counts": {n: d[n]["tp"] + d[n]["fn"] for n in CLASS_NAMES}}
    rep["f1_by_source"] = by_src
    return rep


def save_predictions(records: list[dict], path: Path):
    ser = [{k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in r.items()} for r in records]
    Path(path).write_text(json.dumps(ser))


def load_predictions(path: Path) -> list[dict]:
    recs = json.loads(Path(path).read_text())
    for r in recs:
        for k in ("gt_boxes", "boxes"):
            r[k] = np.asarray(r[k], dtype=float).reshape(-1, 4)
        for k in ("gt_cls", "cls"):
            r[k] = np.asarray(r[k], dtype=int)
        r["scores"] = np.asarray(r["scores"], dtype=float)
    return recs
