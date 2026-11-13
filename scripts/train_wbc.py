"""Классификатор типов лейкоцитов (5 классов PBC).

1. Baseline: морфологические признаки → логистическая регрессия и случайный лес.
2. CNN: YOLO11n-cls, предобученная на ImageNet, дообучение на вырезанных клетках.
Выбор модели — по val, итоговые цифры — на test (тот же сплит, что у детектора).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from hemascope.env import device  # noqa: E402
from hemascope.wbc_features import features  # noqa: E402

DEVICE = device()

CROPS = ROOT / "data" / "wbc_crops"
OUT = ROOT / "reports" / "wbc"
CLASSES = ["BA", "BNE", "EO", "LY", "MO"]  # алфавитный порядок = порядок ImageFolder


def load(split):
    X, y, names = [], [], []
    for k, c in enumerate(CLASSES):
        for p in sorted((CROPS / split / c).glob("*.png")):
            X.append(features(np.asarray(Image.open(p).convert("RGB"))))
            y.append(k)
            names.append(p.name)
    return np.array(X), np.array(y), names


def scores(y, pred):
    return {"accuracy": float(accuracy_score(y, pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
            "macro_f1": float(f1_score(y, pred, average="macro")),
            "confusion": confusion_matrix(y, pred, labels=range(len(CLASSES))).tolist()}


def boot_acc(y, pred, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    acc = [(y[i] == pred[i]).mean() for i in (rng.integers(0, len(y), len(y)) for _ in range(n))]
    return [float(np.quantile(acc, 0.025)), float(np.quantile(acc, 0.975))]


def baselines():
    Xtr, ytr, _ = load("train")
    Xva, yva, _ = load("val")
    Xte, yte, names = load("test")
    res = {}
    models = {
        "features_logreg": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0)),
        "features_random_forest": RandomForestClassifier(n_estimators=500, random_state=0, min_samples_leaf=2),
    }
    for name, m in models.items():
        m.fit(Xtr, ytr)
        pv, pt = m.predict(Xva), m.predict(Xte)
        res[name] = {"val": scores(yva, pv), "test": scores(yte, pt), "test_acc_ci95": boot_acc(yte, pt)}
        print(name, "val acc", round(res[name]["val"]["accuracy"], 3), "test acc", round(res[name]["test"]["accuracy"], 3))
    return res


def cnn(epochs: int, imgsz: int):
    from ultralytics import YOLO

    t0 = time.time()
    model = YOLO(str(ROOT / "weights" / "yolo11n-cls.pt"))
    model.train(data=str(CROPS), epochs=epochs, imgsz=imgsz, batch=32, device=DEVICE, workers=2,
                seed=0, deterministic=True, project=str(ROOT / "runs" / "classify"), name="wbc_yolo11n_cls",
                exist_ok=True, patience=0, cos_lr=True, degrees=180, flipud=0.5, fliplr=0.5,
                hsv_h=0.02, hsv_s=0.4, hsv_v=0.3, plots=True)
    train_sec = time.time() - t0
    best = YOLO(str(ROOT / "runs" / "classify" / "wbc_yolo11n_cls" / "weights" / "best.pt"))
    out = {"train_seconds": round(train_sec, 1), "epochs": epochs, "imgsz": imgsz}
    for split in ("val", "test"):
        y, pred, probs, names = [], [], [], []
        for k, c in enumerate(CLASSES):
            for p in sorted((CROPS / split / c).glob("*.png")):
                r = best.predict(str(p), imgsz=imgsz, device=DEVICE, verbose=False)[0]
                # имена классов модели могут идти в своём порядке — сопоставляем по имени
                pr = r.probs.data.cpu().numpy()
                order = [list(r.names.values()).index(cc) for cc in CLASSES]
                pr = pr[order]
                y.append(k); pred.append(int(pr.argmax())); probs.append(pr.tolist()); names.append(p.name)
        y, pred = np.array(y), np.array(pred)
        out[split] = scores(y, pred)
        if split == "test":
            out["test_acc_ci95"] = boot_acc(y, pred)
            out["test_predictions"] = [{"image": n, "true": CLASSES[a], "pred": CLASSES[b], "probs": pr}
                                       for n, a, b, pr in zip(names, y, pred, probs)]
    print("cnn val acc", round(out["val"]["accuracy"], 3), "test acc", round(out["test"]["accuracy"], 3))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--imgsz", type=int, default=160)
    ap.add_argument("--skip-cnn", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    res = {"classes": CLASSES, "baselines": baselines()}
    if not args.skip_cnn:
        res["cnn_yolo11n_cls"] = cnn(args.epochs, args.imgsz)
    (OUT / "wbc_metrics.json").write_text(json.dumps(res, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
