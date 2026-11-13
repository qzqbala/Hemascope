"""EDA датасета TXL-PBC: размеры, классы, источники, размеры клеток, примеры."""
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
from hemascope.data import CLASS_NAMES, SPLITS, load_split, source_map  # noqa: E402
from hemascope.plotting import COLORS, style  # noqa: E402

OUT = ROOT / "reports" / "eda"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    style()
    meta = source_map().set_index("image")
    rows, boxes = [], []
    for s in SPLITS:
        for smp in load_split(s):
            with Image.open(smp.image) as im:
                w, h = im.size
            src = meta.loc[smp.image.name, "source"]
            row = {"image": smp.image.name, "split": s, "source": src, "w": w, "h": h}
            for i, n in enumerate(CLASS_NAMES):
                row[n] = int((smp.classes == i).sum())
            rows.append(row)
            for b, c in zip(smp.boxes, smp.classes):
                boxes.append({"source": src, "cls": CLASS_NAMES[c],
                              "rel_size": float(np.sqrt((b[2] - b[0]) * (b[3] - b[1]) / (w * h)))})
    df, bx = pd.DataFrame(rows), pd.DataFrame(boxes)
    df.to_csv(OUT / "images.csv", index=False)

    summary = {
        "images": len(df),
        "boxes": int(df[CLASS_NAMES].sum().sum()),
        "per_split_images": df.split.value_counts().to_dict(),
        "per_class_boxes": {n: int(df[n].sum()) for n in CLASS_NAMES},
        "per_source_images": df.source.value_counts().to_dict(),
        "images_without_platelets": int((df["Platelets"] == 0).sum()),
        "boxes_per_image_mean": float(df[CLASS_NAMES].sum(axis=1).mean()),
        "image_sizes": (df["w"].astype(str) + "x" + df["h"].astype(str)).value_counts().head(6).to_dict(),
        "source_by_split": pd.crosstab(df.source, df.split).to_dict(),
        "mean_counts_by_source": df.groupby("source")[CLASS_NAMES].mean().round(2).to_dict(),
    }
    (OUT / "eda_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))

    # 1. клетки по классам и сплитам
    fig, ax = plt.subplots(figsize=(7, 4))
    tab = df.groupby("split")[CLASS_NAMES].sum().loc[list(SPLITS)]
    x = np.arange(len(CLASS_NAMES))
    for k, s in enumerate(SPLITS):
        ax.bar(x + (k - 1) * 0.27, tab.loc[s].values, 0.27, label=s, color=COLORS[k])
    ax.set_xticks(x, ["Лейкоциты", "Эритроциты", "Тромбоциты"])
    ax.set_ylabel("Число размеченных клеток")
    ax.set_title("Дисбаланс классов: эритроцитов в 6 раз больше остальных", loc="left")
    ax.legend(frameon=False)
    fig.savefig(OUT / "class_counts.png")

    # 2. источники
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ct = pd.crosstab(df.source, df.split)[list(SPLITS)]
    left = np.zeros(len(ct))
    for k, s in enumerate(SPLITS):
        ax.barh(ct.index, ct[s], left=left, color=COLORS[k], label=s)
        left += ct[s].values
    ax.set_xlabel("Снимков")
    ax.set_title("Четыре источника, все есть в каждом сплите", loc="left")
    ax.legend(frameon=False, ncol=3, loc="lower right")
    fig.savefig(OUT / "sources.png")

    # 3. клеток на снимок по источникам
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6), sharey=False)
    for ax, n, c in zip(axes, CLASS_NAMES, COLORS):
        data = [df.loc[df.source == s, n].values for s in sorted(df.source.unique())]
        ax.boxplot(data, tick_labels=sorted(df.source.unique()), showfliers=False,
                   medianprops={"color": c, "linewidth": 2})
        ax.set_title({"WBC": "Лейкоцитов", "RBC": "Эритроцитов", "Platelets": "Тромбоцитов"}[n]
                     + " на снимок", loc="left")
        ax.tick_params(axis="x", rotation=30)
    fig.savefig(OUT / "counts_per_image.png")

    # 4. относительный размер клеток
    fig, ax = plt.subplots(figsize=(7, 4))
    for n, c in zip(CLASS_NAMES, COLORS):
        ax.hist(bx.loc[bx.cls == n, "rel_size"], bins=60, alpha=0.65, color=c,
                label={"WBC": "Лейкоциты", "RBC": "Эритроциты", "Platelets": "Тромбоциты"}[n])
    ax.set_xlabel("Размер клетки / размер снимка (√площади)")
    ax.set_ylabel("Клеток")
    ax.set_title("Тромбоциты мелкие: ~2–4% стороны снимка", loc="left")
    ax.legend(frameon=False)
    fig.savefig(OUT / "cell_sizes.png")

    # 5. примеры с разметкой — по одному из каждого источника
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.8))
    for ax, s in zip(axes, sorted(df.source.unique())):
        name = df[(df.source == s) & (df.split == "train")].iloc[3]["image"]
        smp = next(x for x in load_split("train") if x.image.name == name)
        ax.imshow(Image.open(smp.image).convert("RGB"))
        for b, c in zip(smp.boxes, smp.classes):
            ax.add_patch(plt.Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1], fill=False,
                                       lw=1.2, ec=COLORS[c]))
        ax.set_title(s, loc="left")
        ax.axis("off")
    fig.savefig(OUT / "samples.png")
    print(json.dumps(summary, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
