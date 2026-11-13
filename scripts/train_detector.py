"""Обучение детектора клеток крови (YOLO11) на TXL-PBC.

Примеры:
    python scripts/train_detector.py --name yolo11n_pretrained --weights weights/yolo11n.pt
    python scripts/train_detector.py --name yolo11n_scratch --weights yolo11n.yaml   # без предобучения
"""
import argparse
import json
import sys
import time
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--weights", required=True, help="веса .pt (предобучение) или .yaml (с нуля)")
    ap.add_argument("--data", default="full", help="full | loso_raabin | путь к своему yaml")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--imgsz", type=int, default=416)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None, help="по умолчанию — GPU, если есть, иначе CPU")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    sys.path.insert(0, str(ROOT / "src"))
    from hemascope.env import data_yaml, device
    args.device = args.device or device()
    data_path = str(data_yaml(args.data))

    run_dir = ROOT / "runs" / "detect" / args.name
    t0 = time.time()
    if args.resume and (run_dir / "weights" / "last.pt").exists():
        model = YOLO(str(run_dir / "weights" / "last.pt"))
        model.train(resume=True)
    else:
        weights = args.weights
        if weights.endswith(".pt") and not Path(weights).is_absolute():
            weights = str(ROOT / weights)
        model = YOLO(weights)
        model.train(
            data=data_path, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
            device=args.device, workers=2, seed=args.seed, deterministic=True,
            project=str(ROOT / "runs" / "detect"), name=args.name, exist_ok=True,
            patience=0, cos_lr=True, close_mosaic=max(5, args.epochs // 10),
            # аугментации под микроскопию: поворот и отражения допустимы, цвет окраски слегка варьируем
            degrees=10, flipud=0.5, fliplr=0.5, hsv_h=0.02, hsv_s=0.5, hsv_v=0.3,
            pretrained=args.weights.endswith(".pt"),
        )
    meta = {"name": args.name, "weights_init": args.weights, "data": Path(args.data).name, "epochs": args.epochs,
            "imgsz": args.imgsz, "batch": args.batch, "seed": args.seed, "device": args.device,
            "train_seconds": round(time.time() - t0, 1)}
    (run_dir / "train_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
