"""Сводная таблица и графики по всем экспериментам (только чтение готовых предсказаний)."""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from hemascope.data import CLASS_NAMES, DATA_DIR, source_map  # noqa: E402
from hemascope.evaluate import load_predictions  # noqa: E402
from hemascope.metrics import match  # noqa: E402
from hemascope.plotting import COLORS, style  # noqa: E402

OUT = ROOT / "reports" / "results"
RU = {"WBC": "Лейкоциты", "RBC": "Эритроциты", "Platelets": "Тромбоциты"}
METHODS = [  # ключ, подпись, путь к метрикам, путь к предсказаниям test
    ("baseline", "Классический CV (Оцу + Хаф)", "reports/baseline/baseline_metrics.json", "reports/baseline/preds_test.json"),
    ("yolo11n_scratch", "YOLO11n с нуля", "reports/detector/yolo11n_scratch/metrics.json",
     "reports/detector/yolo11n_scratch/preds_test.json"),
    ("yolo11n_pretrained", "YOLO11n, перенос с COCO", "reports/detector/yolo11n_pretrained/metrics.json",
     "reports/detector/yolo11n_pretrained/preds_test.json"),
    ("yolo11s_pretrained", "YOLO11s, перенос с COCO", "reports/detector/yolo11s_pretrained/metrics.json",
     "reports/detector/yolo11s_pretrained/preds_test.json"),
    ("yolo11n_pretrained_512", "YOLO11n, перенос, 512 px", "reports/detector/yolo11n_pretrained_512/metrics.json",
     "reports/detector/yolo11n_pretrained_512/preds_test.json"),
]
T_975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776}
FINAL = "yolo11n_pretrained"


def load_methods():
    out = []
    for key, label, mpath, ppath in METHODS:
        if (ROOT / mpath).exists():
            out.append((key, label, json.loads((ROOT / mpath).read_text()), load_predictions(ROOT / ppath)))
    return out


def results_table(methods):
    rows = []
    for key, label, m, _ in methods:
        t = m["test"]
        row = {"Метод": label,
               "mAP50 (test)": m.get("map_test", {}).get("mAP50"),
               "mAP50-95 (test)": m.get("map_test", {}).get("mAP50_95"),
               "Macro F1": t["macro_f1"]}
        for n in CLASS_NAMES:
            row[f"F1 {RU[n]}"] = t["detection"][n]["f1"]
        for n in CLASS_NAMES:
            ce = t["counting"][n]
            row[f"MAE {RU[n]}"] = ce["mae"]
        row["Время, мс (CPU)"] = m.get("latency_ms_median_cpu")
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "results.csv", index=False)
    return df


def fmt(v, d=3):
    return "—" if v is None or (isinstance(v, float) and np.isnan(v)) else (f"{v:.{d}f}" if isinstance(v, float) else str(v))


def chart_f1(methods):
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(CLASS_NAMES))
    w = 0.8 / len(methods)
    for k, (key, label, m, _) in enumerate(methods):
        vals = [m["test"]["detection"][n]["f1"] for n in CLASS_NAMES]
        bars = ax.bar(x + (k - (len(methods) - 1) / 2) * w, vals, w, label=label,
                      color=["#A0AEC0", "#C9A227", COLORS[0], COLORS[2], COLORS[3]][k % 5])
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.2f}", ha="center", fontsize=8)
    ax.set_xticks(x, [RU[n] for n in CLASS_NAMES])
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("F1 при IoU ≥ 0,5 (test)")
    ax.set_title("Нейросеть опережает классический подход по всем трём типам клеток", loc="left")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3)
    fig.savefig(OUT / "f1_by_class.png")


def chart_learning(methods):
    fig, ax = plt.subplots(figsize=(7, 4))
    for key, label, color in [("yolo11n_scratch", "С нуля", "#C9A227"), ("yolo11n_pretrained", "Перенос с COCO", COLORS[0])]:
        p = ROOT / "runs" / "detect" / key / "results.csv"
        if p.exists():
            r = pd.read_csv(p)
            ax.plot(r["epoch"], r["metrics/mAP50-95(B)"], color=color, lw=2, label=label)
    ax.set_xlabel("Эпоха")
    ax.set_ylabel("mAP50-95 на валидации")
    ax.set_title("Предобучение ускоряет сходимость", loc="left")
    ax.legend(frameon=False)
    fig.savefig(OUT / "learning_curves.png")


def chart_counts(final_recs):
    """Bland–Altman по эритроцитам + истинное vs предсказанное число для всех классов."""
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.6))
    for ax, (c, n) in zip(axes[:3], enumerate(CLASS_NAMES)):
        t = np.array([(r["gt_cls"] == c).sum() for r in final_recs])
        p = np.array([(r["cls"] == c).sum() for r in final_recs])
        jit = np.random.default_rng(0).normal(0, 0.12, (2, len(t)))
        ax.scatter(t + jit[0], p + jit[1], s=12, alpha=0.6, color=COLORS[c])
        lim = max(t.max(), p.max()) + 1
        ax.plot([0, lim], [0, lim], color="#4A5568", lw=1, ls="--")
        ax.set_title(f"{RU[n]}: точно в {np.mean(t == p):.0%} снимков", loc="left")
        ax.set_xlabel("Размечено")
        ax.set_ylabel("Найдено моделью")
    t = np.array([(r["gt_cls"] == 1).sum() for r in final_recs])
    p = np.array([(r["cls"] == 1).sum() for r in final_recs])
    mean, diff = (t + p) / 2, p - t
    md, sd = diff.mean(), diff.std(ddof=1)
    ax = axes[3]
    ax.scatter(mean, diff, s=12, alpha=0.6, color=COLORS[1])
    for v, ls in [(md, "-"), (md + 1.96 * sd, "--"), (md - 1.96 * sd, "--")]:
        ax.axhline(v, color="#4A5568", lw=1, ls=ls)
    ax.text(mean.max(), md + 1.96 * sd, f" +1,96σ = {md + 1.96 * sd:.1f}", va="bottom", ha="right", fontsize=8)
    ax.text(mean.max(), md - 1.96 * sd, f" −1,96σ = {md - 1.96 * sd:.1f}", va="top", ha="right", fontsize=8)
    ax.set_title(f"Bland–Altman, эритроциты: смещение {md:+.2f}", loc="left")
    ax.set_xlabel("Среднее (разметка, модель)")
    ax.set_ylabel("Модель − разметка")
    fig.savefig(OUT / "counting.png")
    return {"rbc_bland_altman": {"bias": float(md), "loa_low": float(md - 1.96 * sd), "loa_high": float(md + 1.96 * sd)}}


def chart_sources(methods):
    rows = []
    for key, label, m, _ in methods:
        for s, v in m["test"]["f1_by_source"].items():
            for n in CLASS_NAMES:
                rows.append({"method": label, "source": f"{s} (n={v['n_images']})", "cls": RU[n], "f1": v[n]})
    df = pd.DataFrame(rows)
    final_label = [l for k, l, _, _ in methods if k == FINAL][0]
    piv = df[df.method == final_label].pivot(index="source", columns="cls", values="f1")[[RU[n] for n in CLASS_NAMES]]
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    im = ax.imshow(piv.values, cmap="Purples", vmin=0, vmax=1)
    ax.set_xticks(range(piv.shape[1]), piv.columns)
    ax.set_yticks(range(piv.shape[0]), piv.index)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.values[i, j]
            ax.text(j, i, "—" if np.isnan(v) else f"{v:.2f}", ha="center", va="center",
                    color="white" if v > 0.6 else "#13232E")
    ax.grid(False)
    ax.set_title("F1 финальной модели по источникам (test)", loc="left")
    fig.colorbar(im, ax=ax, fraction=0.04)
    fig.savefig(OUT / "f1_by_source.png")
    return piv


def error_gallery(final_recs, k=6):
    """Снимки с наибольшей ошибкой подсчёта: зелёный — найдено верно, красный — лишнее, жёлтый — пропуск."""
    err = [sum(abs(int((r["cls"] == c).sum()) - int((r["gt_cls"] == c).sum())) for c in range(3)) for r in final_recs]
    worst = np.argsort(err)[::-1][:k]
    src = source_map().set_index("image")["source"]
    fig, axes = plt.subplots(2, k // 2, figsize=(4.2 * k // 2, 8))
    rows = []
    for ax, i in zip(axes.ravel(), worst):
        r = final_recs[i]
        ax.imshow(Image.open(DATA_DIR / "images" / "test" / r["image"]).convert("RGB"))
        notes = []
        for c, n in enumerate(CLASS_NAMES):
            pm, gm = r["cls"] == c, r["gt_cls"] == c
            _, fp, fn, used, is_tp = match(r["boxes"][pm], r["scores"][pm], r["gt_boxes"][gm])
            for b, ok in zip(r["boxes"][pm], is_tp):
                ax.add_patch(plt.Rectangle(b[:2], b[2] - b[0], b[3] - b[1], fill=False, lw=1.3,
                                           ec="#2F855A" if ok else "#E53E3E"))
            for b, u in zip(r["gt_boxes"][gm], used):
                if not u:
                    ax.add_patch(plt.Rectangle(b[:2], b[2] - b[0], b[3] - b[1], fill=False, lw=1.6,
                                               ec="#D69E2E", ls="--"))
            if fp or fn:
                notes.append(f"{RU[n][:4]}.: +{fp}/−{fn}")
        ax.set_title(f"{src.get(r['image'], '?')} · " + ", ".join(notes), loc="left", fontsize=9)
        ax.axis("off")
        rows.append({"image": r["image"], "source": src.get(r["image"]), "abs_count_error": err[i], "notes": "; ".join(notes)})
    fig.suptitle("Худшие снимки test: зелёный — верно, красный — лишнее, жёлтый пунктир — пропуск",
                 x=0.01, ha="left", fontweight="bold")
    fig.savefig(OUT / "error_gallery.png")
    pd.DataFrame(rows).to_csv(OUT / "error_gallery.csv", index=False)


def chart_wbc():
    p = ROOT / "reports" / "wbc" / "wbc_metrics.json"
    if not p.exists():
        return None
    w = json.loads(p.read_text())
    names = {"BA": "Базофил", "BNE": "Палочкояд.\nнейтрофил", "EO": "Эозинофил", "LY": "Лимфоцит", "MO": "Моноцит"}
    best = w.get("cnn_yolo11n_cls") or w["baselines"]["features_logreg"]
    cm = np.array(best["test"]["confusion"])
    fig, ax = plt.subplots(figsize=(5.6, 4.6))
    ax.imshow(cm, cmap="Purples")
    lab = [names[c] for c in w["classes"]]
    ax.set_xticks(range(5), lab, fontsize=8)
    ax.set_yticks(range(5), lab, fontsize=8)
    for i in range(5):
        for j in range(5):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "#13232E")
    ax.set_xlabel("Предсказано")
    ax.set_ylabel("Истинный тип")
    ax.grid(False)
    ax.set_title(f"Типы лейкоцитов, test: точность {best['test']['accuracy']:.0%} (n={cm.sum()})", loc="left")
    fig.savefig(OUT / "wbc_confusion.png")
    return w


def edge_rule_table(methods):
    rows = []
    for key, label, m, _ in methods:
        if "test_without_edge_rule" not in m:
            continue
        for tag, rep in (("без правила", m["test_without_edge_rule"]), ("с правилом", m["test"])):
            d, c = rep["detection"]["RBC"], rep["counting"]["RBC"]
            rows.append({"Метод": label, "Вариант": tag, "Precision RBC": d["precision"], "Recall RBC": d["recall"],
                         "F1 RBC": d["f1"], "MAE RBC": c["mae"], "Смещение RBC": c["bias"]})
    return pd.DataFrame(rows)


def seeds_table():
    names = ["yolo11n_pretrained", "yolo11n_pretrained_s1", "yolo11n_pretrained_s2"]
    rows = []
    for n in names:
        p = ROOT / "reports" / "detector" / n / "metrics.json"
        if p.exists():
            m = json.loads(p.read_text())
            rows.append({"run": n, "seed": m.get("train", {}).get("seed", 0), "mAP50": m["map_test"]["mAP50"],
                         "mAP50_95": m["map_test"]["mAP50_95"], "macro_f1": m["test"]["macro_f1"],
                         "rbc_mae": m["test"]["counting"]["RBC"]["mae"]})
    if len(rows) < 2:
        return None, None
    df = pd.DataFrame(rows)
    summ = {}
    for col in ("mAP50", "mAP50_95", "macro_f1", "rbc_mae"):
        v = df[col].values
        half = T_975[len(v) - 1] * v.std(ddof=1) / np.sqrt(len(v))
        summ[col] = {"mean": float(v.mean()), "sd": float(v.std(ddof=1)), "ci95": [float(v.mean() - half), float(v.mean() + half)]}
    return df, summ


def loso_chart():
    """Модель без Raabin-WBC против модели, видевшей все источники: F1 на test по источникам."""
    a = ROOT / "reports" / "detector" / "yolo11n_pretrained" / "metrics.json"
    b = ROOT / "reports" / "detector" / "yolo11n_loso_raabin" / "metrics.json"
    if not (a.exists() and b.exists()):
        return None
    A, B = json.loads(a.read_text())["test"]["f1_by_source"], json.loads(b.read_text())["test"]["f1_by_source"]
    out = {}
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    for ax, cls in zip(axes, ["RBC", "WBC"]):
        srcs = list(A)
        x = np.arange(len(srcs))
        va, vb = [A[s][cls] for s in srcs], [B[s][cls] for s in srcs]
        ax.bar(x - 0.2, va, 0.4, color=COLORS[0], label="обучена на всех 4 источниках")
        ax.bar(x + 0.2, vb, 0.4, color="#C9A227", label="Raabin-WBC не видела")
        for i, (p, q) in enumerate(zip(va, vb)):
            if p is not None and q is not None:
                ax.text(i + 0.2, q + 0.01, f"{q - p:+.2f}", ha="center", fontsize=8)
        ax.set_xticks(x, [s.replace("Raabin-WBC", "Raabin-WBC\n(новый источник)") for s in srcs], fontsize=8)
        ax.set_ylim(0, 1.1)
        ax.set_title(f"F1 {RU[cls].lower()} по источникам (test)", loc="left")
        out[cls] = {s: {"all_sources": A[s][cls], "loso": B[s][cls]} for s in srcs}
    axes[0].legend(frameon=False, fontsize=8, loc="lower left")
    fig.savefig(OUT / "loso_raabin.png")
    return out


def write_md(df, extra, wbc, piv):
    lines = ["# Результаты HemaScope (test, 126 снимков)", "",
             "Сгенерировано `scripts/analyze.py`. Пороги уверенности для нейросетей подобраны на val; "
             "параметры классического baseline — по сетке на train. F1 — при IoU ≥ 0,5; MAE — средняя "
             "абсолютная ошибка числа клеток на снимок.", ""]
    cols = list(df.columns)
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("|" + "---|" * len(cols))
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(fmt(r[c], 1 if "мс" in c else 3) for c in cols) + " |")
    lines += ["", "## Подсчёт эритроцитов: Bland–Altman (финальная модель)", "",
              f"Смещение {extra['rbc_bland_altman']['bias']:+.2f} клетки на снимок, пределы согласия "
              f"[{extra['rbc_bland_altman']['loa_low']:.1f}; {extra['rbc_bland_altman']['loa_high']:.1f}].", "",
              "## F1 финальной модели по источникам", "", piv.round(3).to_markdown(), ""]
    if extra.get("edge_rule_table") is not None and len(extra["edge_rule_table"]):
        et = extra["edge_rule_table"]
        lines += ["## Правило краевых клеток (протокол разметки: «у края — только если видно больше половины»)", "",
                  et.round(3).to_markdown(index=False), ""]
    if extra.get("seeds") is not None:
        sd, summ = extra["seeds"]
        lines += ["## Устойчивость к seed (YOLO11n, перенос, 3 запуска)", "", sd.round(4).to_markdown(index=False), ""]
        for k, v in summ.items():
            lines.append(f"- {k}: {v['mean']:.4f} ± {v['sd']:.4f}, 95% ДИ [{v['ci95'][0]:.4f}; {v['ci95'][1]:.4f}]")
        lines.append("")
    if extra.get("loso"):
        lines += ["## Новый источник: модель без Raabin-WBC", "", "![LOSO](loso_raabin.png)", ""]
        for cls, d in extra["loso"].items():
            r = d.get("Raabin-WBC", {})
            lines.append(f"- {RU[cls]}, Raabin-WBC: F1 {r.get('all_sources')} → {r.get('loso')}")
        lines.append("")
    if wbc:
        lines += ["## Классификатор типов лейкоцитов (PBC, 5 классов, test n=50)", "",
                  "| Модель | Точность val | Точность test | 95% ДИ test | Macro F1 test |", "|---|---|---|---|---|"]
        for k, v in list(wbc["baselines"].items()) + ([("cnn_yolo11n_cls", wbc["cnn_yolo11n_cls"])] if "cnn_yolo11n_cls" in wbc else []):
            ci = v.get("test_acc_ci95", [np.nan, np.nan])
            lines.append(f"| {k} | {v['val']['accuracy']:.3f} | {v['test']['accuracy']:.3f} | "
                         f"[{ci[0]:.2f}; {ci[1]:.2f}] | {v['test']['macro_f1']:.3f} |")
    lines += ["", "![F1](f1_by_class.png)", "", "![Подсчёт](counting.png)", "", "![Источники](f1_by_source.png)",
              "", "![Кривые обучения](learning_curves.png)", "", "![Ошибки](error_gallery.png)"]
    (OUT / "results.md").write_text(
        "\n".join(lines),
        encoding="utf-8"
    )


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    style()
    methods = load_methods()
    df = results_table(methods)
    chart_f1(methods)
    chart_learning(methods)
    final = [r for k, _, _, r in methods if k == FINAL][0]
    extra = chart_counts(final)
    piv = chart_sources(methods)
    error_gallery(final)
    wbc = chart_wbc()
    extra["edge_rule_table"] = edge_rule_table(methods)
    extra["seeds"] = seeds_table() if seeds_table()[0] is not None else None
    extra["loso"] = loso_chart()
    ser = {k: v for k, v in extra.items() if k not in ("edge_rule_table", "seeds")}
    if extra["seeds"] is not None:
        ser["seeds"] = extra["seeds"][1]
    (OUT / "extra.json").write_text(
        json.dumps(ser, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )
    write_md(df, extra, wbc, piv)
    print(df.to_string())
    print(extra)


if __name__ == "__main__":
    main()
