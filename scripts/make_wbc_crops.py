"""Вырезает лейкоциты из снимков PBC (тип клетки известен по имени исходного файла).

Результат — папки data/wbc_crops/{train,val,test}/{BA,EO,LY,MO,BNE}/*.png
в том же разбиении, что и детектор: снимок из test детектора не попадёт в train классификатора.
"""
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from hemascope.data import SPLITS, load_split, source_map  # noqa: E402

OUT = ROOT / "data" / "wbc_crops"
PAD = 0.10  # запас вокруг бокса, доля стороны


def crop_box(img: Image.Image, box, pad=PAD) -> Image.Image:
    x1, y1, x2, y2 = box
    side = max(x2 - x1, y2 - y1) * (1 + 2 * pad)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    return img.crop((int(cx - side / 2), int(cy - side / 2), int(cx + side / 2), int(cy + side / 2)))


def main():
    meta = source_map().set_index("image")
    n = 0
    for split in SPLITS:
        for s in load_split(split):
            row = meta.loc[s.image.name]
            if row["source"] != "PBC" or not isinstance(row["wbc_type"], str):
                continue
            wbc = s.boxes[s.classes == 0]
            if len(wbc) == 0:
                continue
            img = Image.open(s.image).convert("RGB")
            # центральный лейкоцит — тот, ради которого делался снимок
            c = np.array(img.size) / 2
            centers = (wbc[:, :2] + wbc[:, 2:]) / 2
            box = wbc[np.argmin(((centers - c) ** 2).sum(1))]
            d = OUT / split / row["wbc_type"]
            d.mkdir(parents=True, exist_ok=True)
            crop_box(img, box).save(d / s.image.name)
            n += 1
    print("crops:", n)
    for split in SPLITS:
        print(split, {p.name: len(list(p.glob("*.png"))) for p in sorted((OUT / split).iterdir())})


if __name__ == "__main__":
    main()
