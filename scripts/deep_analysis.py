"""Углублённый анализ (по мотивам «Limitations and Future Work» статьи BloodContourNet, Diagnostics 2026):

1. Проверка почти-дубликатов train↔test: perceptual hash, SSIM, косинус признаков CNN (ImageNet).
2. Собственный mAP + bootstrap-интервал по снимкам test (сверка со штатным валидатором).
3. Полнота по плотности клеток: изолированные vs «слипшиеся» эритроциты.
4. Полнота по размеру тромбоцитов.
5. Калибровка уверенности детектора (reliability diagram, ECE).
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
from hemascope.ap import bootstrap_map, map_from_cache, precompute  # noqa: E402
from hemascope.data import CLASS_NAMES, load_split  # noqa: E402
from hemascope.evaluate import load_predictions  # noqa: E402
from hemascope.metrics import match  # noqa: E402
from hemascope.plotting import COLORS, style  # noqa: E402

NAME = "yolo11n_pretrained"
OUT = ROOT / "reports" / "deep"
RU = {"WBC": "Лейкоциты", "RBC": "Эритроциты", "Platelets": "Тромбоциты"}


# ---------- 1. почти-дубликаты
def phash(gray32):
    d = cv2.dct(gray32.astype(np.float32))[:8, :8]
    return (d > np.median(d[1:].ravel())).ravel()


def ssim_batch(a, B):
    """SSIM одного снимка a (H,W) с пачкой B (N,H,W), глобальная версия на гауссовых окнах."""
    C1, C2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    def blur(x):
        return cv2.GaussianBlur(x, (11, 11), 1.5)
    a = a.astype(np.float32)
    mu_a = blur(a)
    sa = blur(a * a) - mu_a ** 2
    out = []
    for b in B:
        b = b.astype(np.float32)
        mu_b = blur(b)
        sb = blur(b * b) - mu_b ** 2
        sab = blur(a * b) - mu_a * mu_b
        m = ((2 * mu_a * mu_b + C1) * (2 * sab + C2)) / ((mu_a ** 2 + mu_b ** 2 + C1) * (sa + sb + C2))
        out.append(float(m.mean()))
    return np.array(out)


def near_duplicates():
    from ultralytics import YOLO

    tr, te = load_split("train"), load_split("test")
    def prep(p, n):
        return cv2.resize(np.asarray(Image.open(p).convert("L")), (n, n), interpolation=cv2.INTER_AREA)
    h_tr = np.array([phash(prep(s.image, 32)) for s in tr])
    g_tr = np.array([prep(s.image, 96) for s in tr])
    emb_model = YOLO(str(ROOT / "weights" / "yolo11n-cls.pt"))
    def emb(paths):
        e = np.array([emb_model.embed(str(p), imgsz=224, verbose=False)[0].cpu().numpy() for p in paths])
        return e / np.linalg.norm(e, axis=1, keepdims=True)
    e_tr, e_te = emb([s.image for s in tr]), emb([s.image for s in te])
    # Сырые признаки YOLO11n-cls почти коллинеарны (косинус ≈ 0,99 у любых двух мазков), поэтому
    # порог 0,97 из статьи (ResNet-50) к ним неприменим. Центрируем признаки по train и сравниваем
    # ближайшего соседа test→train с ближайшим соседом train→train: при утечке первое было бы выше.
    mu = e_tr.mean(0)
    c_tr = (e_tr - mu) / np.linalg.norm(e_tr - mu, axis=1, keepdims=True)
    c_te = (e_te - mu) / np.linalg.norm(e_te - mu, axis=1, keepdims=True)
    tt = c_tr @ c_tr.T
    np.fill_diagonal(tt, -1)
    nn_train = tt.max(1)
    rows = []
    for k, s in enumerate(te):
        ham = (h_tr != phash(prep(s.image, 32))).sum(1)
        ss = ssim_batch(prep(s.image, 96), g_tr)
        rows.append({"image": s.image.name, "min_hamming": int(ham.min()), "max_ssim": float(ss.max()),
                     "nn_centered_cosine": float((c_tr @ c_te[k]).max())})
    nn_test = np.array([r["nn_centered_cosine"] for r in rows])
    flagged = [r for r in rows if r["min_hamming"] <= 5 or r["max_ssim"] >= 0.90]
    return {"criteria": "почти-дубликат: pHash Хэмминг ≤ 5 или SSIM ≥ 0,90 (пороги BloodContourNet)",
            "n_test": len(rows), "n_flagged": len(flagged), "flagged": flagged,
            "min_hamming_overall": int(min(r["min_hamming"] for r in rows)),
            "max_ssim_overall": float(max(r["max_ssim"] for r in rows)),
            "cnn_nn_cosine_test_to_train": {"median": float(np.median(nn_test)), "p95": float(np.percentile(nn_test, 95)),
                                            "max": float(nn_test.max())},
            "cnn_nn_cosine_train_to_train": {"median": float(np.median(nn_train)),
                                             "p95": float(np.percentile(nn_train, 95)), "max": float(nn_train.max())},
            "per_image": rows}


# ---------- 3–4. страты
def density(gt_boxes, gt_cls, i, rho=1.5):
    """Число соседних клеток, чьи центры ближе ρ·√(w·h) (формула (1) из BloodContourNet)."""
    c = (gt_boxes[:, :2] + gt_boxes[:, 2:]) / 2
    s = np.sqrt((gt_boxes[i, 2] - gt_boxes[i, 0]) * (gt_boxes[i, 3] - gt_boxes[i, 1]))
    d = np.linalg.norm(c - c[i], axis=1)
    return int(((d <= rho * s) & (np.arange(len(c)) != i)).sum())


def strata(recs):
    """Полнота эритроцитов по плотности и тромбоцитов по размеру (при порогах с val)."""
    dens_bins = {"0–2 соседа": (0, 2), "3–4": (3, 4), "5–6": (5, 6), "7+": (7, 99)}
    rbc = {k: [0, 0] for k in dens_bins}
    plt_rows = []
    from hemascope.data import DATA_DIR
    from hemascope.postprocess import touches_edge
    for r in recs:
        w, h = Image.open(DATA_DIR / "images" / "test" / r["image"]).size
        edge_gt = touches_edge(r["gt_boxes"], w, h) if len(r["gt_boxes"]) else np.zeros(0, bool)
        for c in (1, 2):
            gm = np.where(r["gt_cls"] == c)[0]
            pm = r["cls"] == c
            _, _, _, used, _ = match(r["boxes"][pm], r["scores"][pm], r["gt_boxes"][gm])
            for j, g in enumerate(gm):
                if c == 1:
                    if edge_gt[g]:  # у края соседей меньше по геометрии — исключаем, чтобы не смешивать эффекты
                        continue
                    d = density(r["gt_boxes"], r["gt_cls"], g)
                    for k, (lo, hi) in dens_bins.items():
                        if lo <= d <= hi:
                            rbc[k][0] += int(used[j]); rbc[k][1] += 1
                else:
                    b = r["gt_boxes"][g]
                    plt_rows.append((np.sqrt((b[2] - b[0]) * (b[3] - b[1])), bool(used[j])))
    rbc_out = {k: {"n": v[1], "recall": v[0] / v[1] if v[1] else None} for k, v in rbc.items()}
    plt_out = {}
    if plt_rows:
        sz = np.array([p[0] for p in plt_rows])
        hit = np.array([p[1] for p in plt_rows])
        qs = np.quantile(sz, [0, 1 / 3, 2 / 3, 1])
        for lo, hi, name in zip(qs[:-1], qs[1:], ["мелкие", "средние", "крупные"]):
            m = (sz >= lo) & (sz <= hi)
            plt_out[name] = {"size_px": [float(lo), float(hi)], "n": int(m.sum()), "recall": float(hit[m].mean())}
    return rbc_out, plt_out


# ---------- 5. калибровка
def calibration(raw_recs, bins=10):
    out = {}
    for c, name in enumerate(CLASS_NAMES):
        conf, ok = [], []
        for r in raw_recs:
            pm, gm = r["cls"] == c, r["gt_cls"] == c
            _, _, _, _, is_tp = match(r["boxes"][pm], r["scores"][pm], r["gt_boxes"][gm])
            conf += r["scores"][pm].tolist()
            ok += is_tp.tolist()
        conf, ok = np.array(conf), np.array(ok, float)
        edges = np.linspace(0.05, 1, bins + 1)
        rows, ece = [], 0.0
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (conf >= lo) & (conf < hi)
            if m.sum():
                rows.append({"conf": float(conf[m].mean()), "precision": float(ok[m].mean()), "n": int(m.sum())})
                ece += m.sum() / len(conf) * abs(conf[m].mean() - ok[m].mean())
        out[name] = {"ece": float(ece), "bins": rows}
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    style()
    det = ROOT / "reports" / "detector" / NAME
    recs = load_predictions(det / "preds_test.json")
    raw = load_predictions(det / "preds_test_raw.json")
    res = {}

    print("near-duplicates …")
    res["near_duplicates"] = near_duplicates()

    print("bootstrap mAP …")
    # Свой расчёт — COCO-стиль: сопоставление по убыванию уверенности. Валидатор Ultralytics сопоставляет
    # по убыванию IoU и даёт более высокие значения; для сравнения со статьями берём его цифры,
    # bootstrap-интервал — для своего (более строгого) расчёта.
    cache = precompute(raw)
    own = map_from_cache(cache)
    res["own_map_test"] = own
    res["own_map_bootstrap_ci95"] = bootstrap_map(cache, n_boot=1000)
    ultra = json.loads((det / "metrics.json").read_text())["map_test"]
    res["ultralytics_map_test"] = {"mAP50": ultra["mAP50"], "mAP50_95": ultra["mAP50_95"]}

    print("strata …")
    res["rbc_recall_by_density"], res["platelet_recall_by_size"] = strata(recs)
    print("calibration …")
    res["calibration"] = calibration(raw)
    (OUT / "deep_analysis.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))
    ax = axes[0]
    k = list(res["rbc_recall_by_density"])
    v = [res["rbc_recall_by_density"][x]["recall"] for x in k]
    n = [res["rbc_recall_by_density"][x]["n"] for x in k]
    bars = ax.bar(k, v, color=COLORS[1])
    for b, val, nn in zip(bars, v, n):
        ax.text(b.get_x() + b.get_width() / 2, val + 0.005, f"{val:.3f}\nn={nn}", ha="center", fontsize=8)
    ax.set_ylim(min(v) - 0.08, 1.03)
    ax.set_title("Эритроциты (не у края): полнота по числу соседей", loc="left")
    ax.set_xlabel("Соседей в радиусе 1,5 размера клетки")
    ax = axes[1]
    p = res["platelet_recall_by_size"]
    if p:
        bars = ax.bar(list(p), [p[x]["recall"] for x in p], color=COLORS[2])
        for b, x in zip(bars, p):
            ax.text(b.get_x() + b.get_width() / 2, p[x]["recall"] + 0.01,
                    f"{p[x]['recall']:.2f}\nn={p[x]['n']}", ha="center", fontsize=8)
        ax.set_ylim(0, 1.15)
    ax.set_title("Тромбоциты: полнота по размеру", loc="left")
    ax = axes[2]
    for c, name in enumerate(CLASS_NAMES):
        b = res["calibration"][name]["bins"]
        ax.plot([x["conf"] for x in b], [x["precision"] for x in b], marker="o", color=COLORS[c],
                label=f"{RU[name]} (ECE {res['calibration'][name]['ece']:.3f})")
    ax.plot([0, 1], [0, 1], ls="--", color="#4A5568", lw=1)
    ax.set_xlabel("Уверенность модели")
    ax.set_ylabel("Доля верных")
    ax.set_title("Калибровка уверенности", loc="left")
    ax.legend(frameon=False, fontsize=8)
    fig.savefig(OUT / "strata_calibration.png")
    show = {k: v for k, v in res.items() if k not in ("calibration",)}
    show["near_duplicates"] = {k: v for k, v in res["near_duplicates"].items() if k != "per_image"}
    print(json.dumps(show, ensure_ascii=False, indent=1)[:4000])


if __name__ == "__main__":
    main()
