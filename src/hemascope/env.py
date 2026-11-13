"""Окружение: устройство (GPU/CPU) и конфиги данных YOLO с абсолютными путями под текущую машину."""
from __future__ import annotations

import os
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "txl-pbc"
NAMES = ["WBC", "RBC", "Platelets"]


def device() -> str:
    """HEMASCOPE_DEVICE=cpu|0|1… переопределяет; иначе первая видеокарта, если есть CUDA."""
    env = os.environ.get("HEMASCOPE_DEVICE")
    if env:
        return env
    try:
        import torch

        return "0" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def _write_yaml(path: Path, root: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"path: {root.as_posix()}\ntrain: images/train\nval: images/val\ntest: images/test\n"
        f"nc: 3\nnames: {NAMES}\n", encoding="utf-8")
    return path


def make_loso(source: str = "Raabin-WBC") -> Path:
    """Копия датасета без одного источника в train/val (test — полный). Копируем, а не делаем
    симлинки, чтобы работало и в Windows."""
    from .data import source_map

    root = ROOT / "data" / f"loso_{source.lower().replace('-', '_')}"
    if (root / "DONE").exists():
        return root
    m = source_map()
    for split in ("train", "val", "test"):
        rows = m[m.split == split]
        if split != "test":
            rows = rows[rows.source != source]
        for sub, ext in (("images", ".png"), ("labels", ".txt")):
            (root / sub / split).mkdir(parents=True, exist_ok=True)
            for img in rows.image:
                stem = Path(img).stem
                src = DATA_DIR / sub / split / (stem + ext)
                if src.exists():
                    shutil.copy(src, root / sub / split / (stem + ext))
    (root / "DONE").write_text("ok")
    return root


def data_yaml(kind: str = "full") -> Path:
    """'full' — весь TXL-PBC; 'loso_raabin' — без Raabin-WBC в train/val; иначе — путь к своему yaml."""
    if kind == "full":
        return _write_yaml(ROOT / "runs" / "data_full.yaml", DATA_DIR)
    if kind == "loso_raabin":
        return _write_yaml(ROOT / "runs" / "data_loso_raabin.yaml", make_loso("Raabin-WBC"))
    return Path(kind)
