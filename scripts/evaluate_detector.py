"""Оценка обученного детектора.

1. mAP50 / mAP50-95 на val и test штатным валидатором Ultralytics.
2. Порог уверенности для каждого класса подбирается на VAL (максимум F1) и без изменений
   применяется к TEST — как порог в DermaTriage.
3. Правило краевых клеток (протокол разметки: клетку у края размечают, только если видно больше
   половины) — доля подбирается на val.
4. На test считаются P/R/F1 при IoU 0,5, ошибки подсчёта (MAE, смещение, доля точных),
   bootstrap-интервалы и разбивка по источникам — теми же функциями, что и для классического baseline.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from hemascope.data import CLASS_NAMES, load_split  # noqa: E402
from hemascope.evaluate import records_to_report, save_predictions  # noqa: E402
from hemascope.env import data_yaml, device  # noqa: E402
from hemascope.metrics import detection_prf  # noqa: E402

DEVICE = device()

CONF_GRID = np.round(np.arange(0.05, 0.91, 0.025), 3)
EDGE_RATIOS = [0.3, 0.35, 0.4, 0.45, 0.5, 0.6]


def edge_rule(recs, split, ratio):
    from PIL import Image

    from hemascope.data import DATA_DIR
    from hemascope.postprocess import edge_filter

    out = []
    for r in recs:
        w, h = Image.open(DATA_DIR / "images" / split / r["image"]).size
        b, c, s = edge_filter(r["boxes"], r["cls"], r["scores"], w, h, ratio)
        out.append({**r, "boxes": b, "cls": c, "scores": s})
    return out


def predict_all(model, samples, imgsz, min_conf=0.05):
    recs, t = [], []
    for s in samples:
        t0 = time.perf_counter()
        r = model.predict(str(s.image), imgsz=imgsz, conf=min_conf, iou=0.6, device=DEVICE,
                          verbose=False, max_det=300)[0]
        t.append(time.perf_counter() - t0)
        b = r.boxes
        recs.append({"image": s.image.name, "gt_boxes": s.boxes, "gt_cls": s.classes,
                     "boxes": b.xyxy.cpu().numpy(), "cls": b.cls.cpu().numpy().astype(int),
                     "scores": b.conf.cpu().numpy()})
    return recs, float(np.median(t) * 1000)


def apply_thresholds(recs, thr: dict):
    out = []
    for r in recs:
        keep = np.array([s >= thr[CLASS_NAMES[c]] for c, s in zip(r["cls"], r["scores"])], dtype=bool)
        out.append({**r, "boxes": r["boxes"][keep], "cls": r["cls"][keep], "scores": r["scores"][keep]})
    return out


def tune_thresholds(val_recs):
    thr, curves = {}, {}
    for c, name in enumerate(CLASS_NAMES):
        f1s = []
        for t in CONF_GRID:
            d = detection_prf(apply_thresholds(val_recs, {n: (t if n == name else 1.1) for n in CLASS_NAMES}))
            f1s.append(d[name]["f1"])
        thr[name] = float(CONF_GRID[int(np.argmax(f1s))])
        curves[name] = [round(x, 4) for x in f1s]
    return thr, curves


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--imgsz", type=int, default=None, help="по умолчанию — как при обучении")
    ap.add_argument("--exclude-val-source", default=None,
                    help="источник, исключённый из обучения (LOSO): не используем его и при подборе порогов")
    args = ap.parse_args()
    from ultralytics import YOLO

    run_dir = ROOT / "runs" / "detect" / args.name
    out_dir = ROOT / "reports" / "detector" / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(run_dir / "weights" / "best.pt"))
    meta_path = run_dir / "train_meta.json"
    if args.imgsz is None:
        args.imgsz = json.loads(meta_path.read_text())["imgsz"] if meta_path.exists() else 416

    res = {"name": args.name, "imgsz": args.imgsz}
    for split in ("val", "test"):
        m = model.val(data=str(data_yaml("full")), split=split, imgsz=args.imgsz,
                      device=DEVICE, batch=16, plots=True, project=str(out_dir), name=f"ultra_{split}",
                      exist_ok=True, verbose=False)
        res[f"map_{split}"] = {
            "mAP50": float(m.box.map50), "mAP50_95": float(m.box.map),
            "per_class_AP50": {CLASS_NAMES[i]: float(m.box.ap50[k]) for k, i in enumerate(m.box.ap_class_index)},
            "per_class_AP50_95": {CLASS_NAMES[i]: float(m.box.ap[k]) for k, i in enumerate(m.box.ap_class_index)},
        }

    val_samples = load_split("val")
    if args.exclude_val_source:
        from hemascope.data import source_map
        src = source_map().set_index("image")["source"]
        val_samples = [s for s in val_samples if src[s.image.name] != args.exclude_val_source]
    res["val_images_for_thresholds"] = len(val_samples)
    val_recs, _ = predict_all(model, val_samples, args.imgsz)
    thr, curves = tune_thresholds(val_recs)
    res["thresholds_from_val"] = thr
    res["val_f1_curves"] = {"conf": CONF_GRID.tolist(), **curves}
    res["val"] = records_to_report(apply_thresholds(val_recs, thr), with_ci=False)

    # правило краевых клеток из протокола разметки: долю подбираем на val по F1 эритроцитов
    val_thr = apply_thresholds(val_recs, thr)
    edge_curve = {r: detection_prf(edge_rule(val_thr, "val", r))["RBC"]["f1"] for r in EDGE_RATIOS}
    ratio = max(edge_curve, key=edge_curve.get)
    res["edge_rule"] = {"ratio_from_val": ratio, "val_rbc_f1_by_ratio": edge_curve,
                        "val_rbc_f1_without_rule": detection_prf(val_thr)["RBC"]["f1"]}

    test_raw, ms = predict_all(model, load_split("test"), args.imgsz)
    test_plain = apply_thresholds(test_raw, thr)
    test = edge_rule(test_plain, "test", ratio)
    res["test_without_edge_rule"] = records_to_report(test_plain, with_ci=False)
    res["test"] = records_to_report(test)
    res["latency_ms_median_cpu"] = ms
    meta = run_dir / "train_meta.json"
    if meta.exists():
        res["train"] = json.loads(meta.read_text())
    save_predictions(test, out_dir / "preds_test.json")
    save_predictions(test_raw, out_dir / "preds_test_raw.json")
    (out_dir / "metrics.json").write_text(json.dumps(res, ensure_ascii=False, indent=2))
    print(json.dumps({k: res[k] for k in ("map_test", "thresholds_from_val")}, ensure_ascii=False, indent=1))
    print("test macro F1", round(res["test"]["macro_f1"], 4), "latency", round(ms, 1))


if __name__ == "__main__":
    main()
