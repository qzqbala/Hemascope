"""Экспорт детектора в ONNX и сравнение скорости/качества с PyTorch на CPU.

Проверяем, что ускорение не испортило ответы: число клеток каждого класса на test должно совпасть.
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


def bench(model, paths, imgsz, thr, warmup=3):
    for p in paths[:warmup]:
        model.predict(str(p), imgsz=imgsz, device="cpu", verbose=False)
    times, counts = [], []
    for p in paths:
        t0 = time.perf_counter()
        r = model.predict(str(p), imgsz=imgsz, conf=min(thr.values()), iou=0.6, device="cpu", verbose=False)[0]
        times.append(time.perf_counter() - t0)
        c, s = r.boxes.cls.cpu().numpy().astype(int), r.boxes.conf.cpu().numpy()
        counts.append([int(((c == k) & (s >= thr[n])).sum()) for k, n in enumerate(CLASS_NAMES)])
    return np.array(times) * 1000, np.array(counts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="yolo11n_pretrained")
    ap.add_argument("--imgsz", type=int, default=None)
    args = ap.parse_args()
    from ultralytics import YOLO

    wdir = ROOT / "runs" / "detect" / args.name / "weights"
    pt = YOLO(str(wdir / "best.pt"))
    if args.imgsz is None:
        args.imgsz = json.loads((ROOT / "reports" / "detector" / args.name / "metrics.json").read_text()).get("imgsz", 416)
    onnx_path = pt.export(format="onnx", imgsz=args.imgsz, simplify=True, dynamic=False)
    met = json.loads((ROOT / "reports" / "detector" / args.name / "metrics.json").read_text())
    thr = met["thresholds_from_val"]
    paths = [s.image for s in load_split("test")]
    t_pt, c_pt = bench(pt, paths, args.imgsz, thr)
    t_ox, c_ox = bench(YOLO(onnx_path, task="detect"), paths, args.imgsz, thr)
    res = {
        "onnx_file": str(Path(onnx_path).relative_to(ROOT)),
        "size_mb": {"pytorch": round((wdir / "best.pt").stat().st_size / 2**20, 2),
                    "onnx": round(Path(onnx_path).stat().st_size / 2**20, 2)},
        "latency_ms_median": {"pytorch": float(np.median(t_pt)), "onnx": float(np.median(t_ox))},
        "latency_ms_p90": {"pytorch": float(np.percentile(t_pt, 90)), "onnx": float(np.percentile(t_ox, 90))},
        "images_with_identical_counts": float((c_pt == c_ox).all(axis=1).mean()),
        "max_count_diff": int(np.abs(c_pt - c_ox).max()),
        "cpu_threads": __import__("torch").get_num_threads(),
    }
    res["speedup"] = res["latency_ms_median"]["pytorch"] / res["latency_ms_median"]["onnx"]
    out = ROOT / "reports" / "detector" / args.name / "onnx_benchmark.json"
    out.write_text(json.dumps(res, ensure_ascii=False, indent=2))
    print(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
