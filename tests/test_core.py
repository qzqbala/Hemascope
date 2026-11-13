"""Быстрые проверки ядра: метрики, разметка, разбиение, проверка качества, baseline."""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hemascope.metrics import count_errors, detection_prf, iou_matrix, match  # noqa: E402

DATA = ROOT / "data" / "txl-pbc"
needs_data = pytest.mark.skipif(not DATA.exists(), reason="датасет не скачан")


def test_iou_identical_and_disjoint():
    a = np.array([[0, 0, 10, 10]], float)
    b = np.array([[0, 0, 10, 10], [20, 20, 30, 30], [5, 0, 15, 10]], float)
    iou = iou_matrix(a, b)[0]
    assert iou[0] == pytest.approx(1.0)
    assert iou[1] == 0
    assert iou[2] == pytest.approx(50 / 150)


def test_match_counts_tp_fp_fn():
    gt = np.array([[0, 0, 10, 10], [20, 20, 30, 30]], float)
    pred = np.array([[0, 0, 10, 10], [0, 0, 10, 11], [50, 50, 60, 60]], float)  # дубль + лишний
    tp, fp, fn, _, _ = match(pred, np.array([0.9, 0.8, 0.7]), gt)
    assert (tp, fp, fn) == (1, 2, 1)


def test_detection_prf_perfect():
    gt = np.array([[0, 0, 10, 10], [20, 20, 30, 30]], float)
    rec = {"gt_boxes": gt, "gt_cls": np.array([0, 1]), "boxes": gt, "cls": np.array([0, 1]),
           "scores": np.array([1.0, 1.0])}
    d = detection_prf([rec])
    assert d["WBC"]["f1"] == 1 and d["RBC"]["f1"] == 1


def test_count_errors():
    e = count_errors([10, 0, 5], [12, 0, 5])
    assert e["mae"] == pytest.approx(2 / 3)
    assert e["exact"] == pytest.approx(2 / 3)
    assert e["bias"] == pytest.approx(2 / 3)


@needs_data
def test_splits_disjoint_and_complete():
    from hemascope.data import source_map

    m = source_map()
    assert len(m) == 1260
    assert m["split"].notna().all()
    assert m["image"].is_unique


@needs_data
def test_labels_inside_image():
    from hemascope.data import load_split

    for s in load_split("test")[:30]:
        assert (s.boxes[:, 2] > s.boxes[:, 0]).all() and (s.boxes[:, 3] > s.boxes[:, 1]).all()
        assert set(np.unique(s.classes)) <= {0, 1, 2}


@needs_data
def test_quality_rejects_blur_and_darkness():
    import cv2
    from PIL import Image

    from hemascope.data import load_split
    from hemascope.pipeline import check_quality

    img = np.asarray(Image.open(load_split("val")[0].image).convert("RGB"))
    assert check_quality(img).ok
    blurred = cv2.GaussianBlur(img, (0, 0), 6 * min(img.shape[:2]) / 400)
    assert not check_quality(blurred, min_sharp=0.018).ok
    assert not check_quality((img * 0.3).astype(np.uint8), min_bright=110).ok


@needs_data
def test_baseline_runs_and_finds_cells():
    from PIL import Image

    from hemascope.baseline import detect
    from hemascope.data import load_split

    s = load_split("val")[0]
    boxes, cls, scores = detect(np.asarray(Image.open(s.image).convert("RGB")))
    assert len(boxes) == len(cls) == len(scores) > 0
    assert (cls == 1).sum() > 0  # эритроциты есть на любом снимке


def test_edge_filter_drops_small_edge_boxes_only():
    from hemascope.postprocess import edge_filter

    full = [[100, 100, 140, 140], [200, 200, 240, 240], [300, 100, 340, 140]]  # целые клетки 40×40
    boxes = np.array(full + [[0, 50, 10, 90],      # у края, видно 1/4 → убрать
                             [0, 150, 30, 190]],   # у края, видно 3/4 → оставить
                     float)
    cls = np.ones(len(boxes), int)
    b, c, s = edge_filter(boxes, cls, np.ones(len(boxes)), 400, 400, ratio=0.5)
    assert len(b) == 4 and not any((b == [0, 50, 10, 90]).all(1))


def test_own_map_perfect_predictions():
    from hemascope.ap import map_from_cache, precompute

    gt = np.array([[0, 0, 10, 10], [20, 20, 30, 30], [40, 40, 44, 44]], float)
    rec = {"gt_boxes": gt, "gt_cls": np.array([0, 1, 2]), "boxes": gt, "cls": np.array([0, 1, 2]),
           "scores": np.array([0.9, 0.9, 0.9])}
    m = map_from_cache(precompute([rec]))
    assert m["mAP50"] == pytest.approx(1.0) and m["mAP50_95"] == pytest.approx(1.0)


def test_duplicate_labels_removed(tmp_path):
    from hemascope.data import read_yolo_labels

    f = tmp_path / "x.txt"
    f.write_text("1 0.5 0.5 0.1 0.1\n1 0.5 0.5 0.1 0.1\n0 0.2 0.2 0.3 0.3\n")
    boxes, cls = read_yolo_labels(f, 100, 100)
    assert len(boxes) == 2 and sorted(cls.tolist()) == [0, 1]
