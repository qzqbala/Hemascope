"""Чтение TXL-PBC: изображения, YOLO-разметка, карта источников."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "txl-pbc"
CLASS_NAMES = ["WBC", "RBC", "Platelets"]
CLASS_RU = {"WBC": "лейкоциты", "RBC": "эритроциты", "Platelets": "тромбоциты"}
SPLITS = ("train", "val", "test")

# типы лейкоцитов в PBC кодируются префиксом исходного имени файла
WBC_TYPES = {"BA": "базофил", "EO": "эозинофил", "LY": "лимфоцит", "MO": "моноцит", "BNE": "палочкоядерный нейтрофил"}


@dataclass
class Sample:
    image: Path
    split: str
    boxes: np.ndarray   # (N, 4) x1, y1, x2, y2 в пикселях
    classes: np.ndarray  # (N,) int


def read_yolo_labels(label_path: Path, width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    if not label_path.exists() or label_path.stat().st_size == 0:
        return np.zeros((0, 4)), np.zeros((0,), dtype=int)
    arr = np.loadtxt(label_path, ndmin=2)
    # в 5 файлах разметки есть точные дубли строк (47 шт.) — убираем, как и YOLO при обучении
    _, keep = np.unique(arr, axis=0, return_index=True)
    arr = arr[np.sort(keep)]
    cls = arr[:, 0].astype(int)
    xc, yc, w, h = arr[:, 1] * width, arr[:, 2] * height, arr[:, 3] * width, arr[:, 4] * height
    boxes = np.stack([xc - w / 2, yc - h / 2, xc + w / 2, yc + h / 2], axis=1)
    return boxes, cls


def load_split(split: str, data_dir: Path = DATA_DIR) -> list[Sample]:
    from PIL import Image

    out = []
    for img in sorted((data_dir / "images" / split).glob("*.png")):
        with Image.open(img) as im:
            w, h = im.size
        boxes, cls = read_yolo_labels(data_dir / "labels" / split / (img.stem + ".txt"), w, h)
        out.append(Sample(img, split, boxes, cls))
    return out


def source_map(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Таблица: новое имя файла → источник (BCCD/BCDD/PBC/Raabin-WBC), сплит, тип лейкоцита для PBC."""
    meta = pd.read_excel(data_dir / "metadata_file.xlsx")
    meta = meta.rename(columns={"New_Image_Name": "image", "Source_Dataset": "source",
                                "Original_Image_Name": "original"})
    split_of = {}
    for s in SPLITS:
        for p in (data_dir / "images" / s).glob("*.png"):
            split_of[p.name] = s
    meta["split"] = meta["image"].map(split_of)
    prefix = meta["original"].str.extract(r"^([A-Z]+)_")[0]
    meta["wbc_type"] = np.where(meta["source"].eq("PBC"), prefix, None)
    return meta[["image", "source", "split", "original", "wbc_type"]]


def counts(classes: np.ndarray) -> dict[str, int]:
    return {name: int((classes == i).sum()) for i, name in enumerate(CLASS_NAMES)}
