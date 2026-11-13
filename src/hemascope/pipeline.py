"""Инференс HemaScope: снимок(и) мазка → проверка качества → детекция → подсчёт → типы лейкоцитов.

Важно: модель считает клетки в поле зрения. Перевести это в концентрацию (клеток/мкл) без калибровки
микроскопа и толщины мазка нельзя, поэтому выводятся соотношения и лейкоцитарная формула,
а не «норма/не норма».
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

from .data import CLASS_NAMES
from .postprocess import edge_filter

ROOT = Path(__file__).resolve().parents[2]
WBC_CLASSES = ["BA", "BNE", "EO", "LY", "MO"]
WBC_RU = {"BA": "Базофил", "BNE": "Палочкоядерный нейтрофил", "EO": "Эозинофил",
          "LY": "Лимфоцит", "MO": "Моноцит"}
WBC_SHORT = {"BA": "Базофил", "BNE": "Нейтрофил п/я", "EO": "Эозинофил", "LY": "Лимфоцит", "MO": "Моноцит"}
BOX_COLORS = {"WBC": (107, 76, 154), "RBC": (200, 66, 59), "Platelets": (31, 138, 138)}


@dataclass
class Quality:
    ok: bool
    sharpness: float
    brightness: float
    problems: list[str] = field(default_factory=list)


def sharpness(gray: np.ndarray) -> float:
    """Доля мелких деталей относительно контраста: |снимок − его размытая копия| / σ.
    Обычная дисперсия лапласиана плохо работает на микроскопии: гладкие, но резкие снимки
    одного источника «размытее» резких снимков другого. Нормировка на контраст это убирает
    (сравнение признаков — в reports/quality/quality_check.json)."""
    g = gray.astype(np.float64)
    return float(np.abs(g - cv2.GaussianBlur(g, (0, 0), 1.5)).mean() / (g.std() + 1e-6))


def check_quality(img: np.ndarray, min_sharp=0.018, min_bright=119, max_bright=220) -> Quality:
    """Резкость и яркость на снимке, приведённом к короткой стороне 400 px.
    Пороги откалиброваны по обучающим снимкам (scripts/calibrate_quality.py)."""
    k = 400 / min(img.shape[:2])
    small = cv2.resize(img, None, fx=k, fy=k, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
    sharp = sharpness(gray)
    bright = float(gray.mean())
    probs = []
    if sharp < min_sharp:
        probs.append("снимок размыт — подстройте фокус микроскопа")
    if bright < min_bright:
        probs.append("снимок слишком тёмный — добавьте освещение")
    if bright > max_bright:
        probs.append("снимок пересвечен — уменьшите освещение")
    return Quality(not probs, sharp, bright, probs)


@dataclass
class FieldResult:
    boxes: np.ndarray
    classes: np.ndarray
    scores: np.ndarray
    wbc_types: list[dict]
    quality: Quality
    annotated: Image.Image

    def counts(self) -> dict[str, int]:
        return {n: int((self.classes == i).sum()) for i, n in enumerate(CLASS_NAMES)}


class HemaScope:
    def __init__(self, detector: str | Path, classifier: str | Path | None = None,
                 thresholds: dict | None = None, imgsz: int = 416, wbc_min_conf: float = 0.6,
                 quality_limits: dict | None = None, edge_ratio: float | None = 0.4):
        from ultralytics import YOLO

        self.det = YOLO(str(detector), task="detect")
        self.cls = YOLO(str(classifier), task="classify") if classifier else None
        self.imgsz = imgsz
        self.thr = thresholds or {n: 0.25 for n in CLASS_NAMES}
        self.wbc_min_conf = wbc_min_conf
        self.quality_limits = quality_limits or {}
        self.edge_ratio = edge_ratio
        from .env import device
        self.device = device()

    @classmethod
    def from_run(cls, det_name="yolo11n_pretrained", with_classifier=True, onnx=False):
        det_dir = ROOT / "runs" / "detect" / det_name / "weights"
        det = det_dir / ("best.onnx" if onnx and (det_dir / "best.onnx").exists() else "best.pt")
        metrics = ROOT / "reports" / "detector" / det_name / "metrics.json"
        m = json.loads(metrics.read_text()) if metrics.exists() else {}
        thr = m.get("thresholds_from_val")
        edge = m.get("edge_rule", {}).get("ratio_from_val", 0.4)
        imgsz = m.get("imgsz", 416)
        clf = ROOT / "runs" / "classify" / "wbc_yolo11n_cls" / "weights" / "best.pt"
        ql = ROOT / "reports" / "quality" / "quality_limits.json"
        return cls(det, clf if with_classifier and clf.exists() else None, thr, imgsz=imgsz,
                   quality_limits=json.loads(ql.read_text()) if ql.exists() else None, edge_ratio=edge)

    def analyze_field(self, image: Image.Image | str | Path) -> FieldResult:
        img = Image.open(image) if not isinstance(image, Image.Image) else image
        img = img.convert("RGB")
        arr = np.asarray(img)
        q = check_quality(arr, **self.quality_limits)
        r = self.det.predict(arr[..., ::-1], imgsz=self.imgsz, conf=min(self.thr.values()), iou=0.6,
                             device=self.device, verbose=False, max_det=300)[0]
        b = r.boxes.xyxy.cpu().numpy()
        c = r.boxes.cls.cpu().numpy().astype(int)
        s = r.boxes.conf.cpu().numpy()
        keep = np.array([sc >= self.thr[CLASS_NAMES[k]] for k, sc in zip(c, s)], dtype=bool)
        b, c, s = b[keep], c[keep], s[keep]
        if self.edge_ratio:
            b, c, s = edge_filter(b, c, s, img.width, img.height, self.edge_ratio)

        wbc_types = []
        if self.cls is not None:
            for box in b[c == 0]:
                crop = _square_crop(img, box)
                pr = self.cls.predict(crop, imgsz=160, device=self.device, verbose=False)[0].probs
                probs = pr.data.cpu().numpy()
                model_names = self.cls.names
                best = int(probs.argmax())
                code = model_names[best]
                wbc_types.append({"box": box.tolist(), "type": code, "type_ru": WBC_RU.get(code, code),
                                  "confidence": float(probs[best]),
                                  "uncertain": bool(probs[best] < self.wbc_min_conf)})
        return FieldResult(b, c, s, wbc_types, q, _draw(img, b, c, s, wbc_types))

    def analyze_smear(self, images: list) -> dict:
        """Несколько полей зрения одного мазка → суммарный подсчёт и лейкоцитарная формула."""
        fields = [self.analyze_field(im) for im in images]
        good = [f for f in fields if f.quality.ok]
        tot = {n: sum(f.counts()[n] for f in good) for n in CLASS_NAMES}
        types = [t for f in good for t in f.wbc_types if not t["uncertain"]]
        formula = {}
        if types:
            for code in WBC_CLASSES:
                formula[WBC_RU[code]] = round(100 * sum(t["type"] == code for t in types) / len(types), 1)
        return {
            "fields_total": len(fields), "fields_used": len(good),
            "counts": tot,
            "per_field_mean": {n: round(tot[n] / max(len(good), 1), 1) for n in CLASS_NAMES},
            "rbc_per_wbc": round(tot["RBC"] / tot["WBC"], 1) if tot["WBC"] else None,
            "wbc_classified": len(types),
            "wbc_uncertain": sum(t["uncertain"] for f in good for t in f.wbc_types),
            "leukocyte_formula_pct": formula,
            "fields": fields,
        }


def _square_crop(img: Image.Image, box, pad=0.10) -> Image.Image:
    x1, y1, x2, y2 = box
    side = max(x2 - x1, y2 - y1) * (1 + 2 * pad)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    return img.crop((int(cx - side / 2), int(cy - side / 2), int(cx + side / 2), int(cy + side / 2)))


def _draw(img: Image.Image, boxes, classes, scores, wbc_types) -> Image.Image:
    from PIL import ImageFont

    out = img.copy()
    d = ImageDraw.Draw(out)
    lw = max(2, round(min(img.size) / 220))
    size = max(12, min(img.size) // 26)
    try:  # DejaVuSans поставляется с matplotlib и содержит кириллицу
        from matplotlib import font_manager
        font = ImageFont.truetype(font_manager.findfont("DejaVu Sans"), size)
    except Exception:
        font = ImageFont.load_default(size=size)
    wbc_iter = iter(wbc_types)
    for box, c, s in zip(boxes, classes, scores):
        name = CLASS_NAMES[c]
        d.rectangle(box.tolist(), outline=BOX_COLORS[name], width=lw)
        if name == "WBC":
            t = next(wbc_iter, None)
            label = (f"{WBC_SHORT.get(t['type'], t['type'])} {t['confidence']:.0%}" + (" ?" if t["uncertain"] else "")
                     if t else f"WBC {s:.0%}")
            x, y = box[0] + lw, box[1] + lw
            tb = d.textbbox((x, y), label, font=font)
            d.rectangle((tb[0] - 2, tb[1] - 2, tb[2] + 2, tb[3] + 2), fill=BOX_COLORS[name])
            d.text((x, y), label, fill=(255, 255, 255), font=font)
    return out
