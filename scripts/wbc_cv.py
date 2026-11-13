"""5-кратная стратифицированная кросс-валидация классификатора лейкоцитов.

Test одного сплита — всего 50 клеток, поэтому точность оцениваем ещё и по всем 500 клеткам:
каждая клетка попадает в тест ровно один раз. Внутри обучающих фолдов 20% отводятся под
валидацию для выбора эпохи — тестовый фолд в выборе не участвует.
"""
import json
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from hemascope.env import device  # noqa: E402
from hemascope.wbc_features import features  # noqa: E402

DEVICE = device()

CROPS = ROOT / "data" / "wbc_crops"
CLASSES = ["BA", "BNE", "EO", "LY", "MO"]
TMP = ROOT / "data" / "wbc_cv_tmp"


def main(epochs=25, imgsz=160, folds=5):
    from ultralytics import YOLO

    paths, y = [], []
    for split in ("train", "val", "test"):
        for k, c in enumerate(CLASSES):
            for p in sorted((CROPS / split / c).glob("*.png")):
                paths.append(p)
                y.append(k)
    paths, y = np.array(paths), np.array(y)
    X = np.array([features(np.asarray(Image.open(p).convert("RGB"))) for p in paths])

    skf = StratifiedKFold(folds, shuffle=True, random_state=0)
    pred_lr, pred_cnn, conf_cnn = np.zeros_like(y), np.zeros_like(y), np.zeros(len(y))
    for f, (tr, te) in enumerate(skf.split(X, y)):
        lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(X[tr], y[tr])
        pred_lr[te] = lr.predict(X[te])

        tr2, va2 = train_test_split(tr, test_size=0.2, stratify=y[tr], random_state=f)
        if TMP.exists():
            shutil.rmtree(TMP)
        for part, idx in (("train", tr2), ("val", va2), ("test", te)):
            for i in idx:
                d = TMP / part / CLASSES[y[i]]
                d.mkdir(parents=True, exist_ok=True)
                shutil.copy(paths[i], d / paths[i].name)
        m = YOLO(str(ROOT / "weights" / "yolo11n-cls.pt"))
        m.train(data=str(TMP), epochs=epochs, imgsz=imgsz, batch=32, device=DEVICE, workers=2, seed=f,
                deterministic=True, project=str(ROOT / "runs" / "classify"), name=f"wbc_cv_fold{f}",
                exist_ok=True, patience=0, cos_lr=True, degrees=180, flipud=0.5, fliplr=0.5,
                hsv_h=0.02, hsv_s=0.4, hsv_v=0.3, plots=False, verbose=False)
        best = YOLO(str(ROOT / "runs" / "classify" / f"wbc_cv_fold{f}" / "weights" / "best.pt"))
        order = [list(best.names.values()).index(c) for c in CLASSES]
        for i in te:
            pr = best.predict(str(paths[i]), imgsz=imgsz, device=DEVICE, verbose=False)[0].probs.data.cpu().numpy()[order]
            pred_cnn[i], conf_cnn[i] = int(pr.argmax()), float(pr.max())
        print(f"fold {f}: LR {accuracy_score(y[te], pred_lr[te]):.3f}  CNN {accuracy_score(y[te], pred_cnn[te]):.3f}")
    shutil.rmtree(TMP, ignore_errors=True)

    def summ(pred):
        accs = [accuracy_score(y[te], pred[te]) for _, te in skf.split(X, y)]
        return {"accuracy": float(accuracy_score(y, pred)), "macro_f1": float(f1_score(y, pred, average="macro")),
                "fold_acc": [float(a) for a in accs], "fold_acc_std": float(np.std(accs)),
                "confusion": confusion_matrix(y, pred).tolist(), "errors": int((pred != y).sum())}

    unc = conf_cnn < 0.6
    res = {"n": int(len(y)), "folds": folds, "classes": CLASSES,
           "features_logreg": summ(pred_lr), "cnn_yolo11n_cls": summ(pred_cnn),
           "cnn_uncertainty_gate_0.6": {
               "flagged": int(unc.sum()),
               "errors_flagged": int((unc & (pred_cnn != y)).sum()),
               "errors_total": int((pred_cnn != y).sum()),
               "accuracy_on_confident": float(accuracy_score(y[~unc], pred_cnn[~unc])) if (~unc).any() else None}}
    out = ROOT / "reports" / "wbc" / "wbc_cv.json"
    out.write_text(json.dumps(res, ensure_ascii=False, indent=2))
    print(json.dumps({k: (v if not isinstance(v, dict) else {a: b for a, b in v.items() if a != "confusion"})
                      for k, v in res.items()}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
