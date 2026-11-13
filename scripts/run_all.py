"""Полный конвейер HemaScope одной командой (Windows / Linux / macOS).

    python scripts/run_all.py --profile gpu     # видеокарта: 640 px, 100 эпох, + YOLO11s
    python scripts/run_all.py --profile cpu     # без видеокарты: 416 px, 30 эпох

Готовые шаги пропускаются (можно прервать и запустить снова — продолжит с места остановки,
прерванное обучение продолжится с последней эпохи). --force переобучает всё заново.
"""
import argparse
import json
import shutil
import subprocess
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
WEIGHTS = ["yolo11n.pt", "yolo11s.pt", "yolo11n-cls.pt"]
DATA_REPO = "https://github.com/lugan113/TXL-PBC_Dataset"

PROFILES = {
    # name: (imgsz, epochs, batch, детекторы)
    "gpu": (640, 100, 16, [
        ("yolo11n_pretrained", "weights/yolo11n.pt", "full", 0),
        ("yolo11n_pretrained_s1", "weights/yolo11n.pt", "full", 1),
        ("yolo11n_pretrained_s2", "weights/yolo11n.pt", "full", 2),
        ("yolo11n_scratch", "yolo11n.yaml", "full", 0),
        ("yolo11n_loso_raabin", "weights/yolo11n.pt", "loso_raabin", 0),
        ("yolo11s_pretrained", "weights/yolo11s.pt", "full", 0),
    ]),
    "cpu": (416, 30, 16, [
        ("yolo11n_pretrained", "weights/yolo11n.pt", "full", 0),
        ("yolo11n_scratch", "yolo11n.yaml", "full", 0),
        ("yolo11n_loso_raabin", "weights/yolo11n.pt", "loso_raabin", 0),
        ("yolo11n_pretrained_s1", "weights/yolo11n.pt", "full", 1),
        ("yolo11n_pretrained_s2", "weights/yolo11n.pt", "full", 2),
    ]),
}


def step(title, args):
    print(f"\n=== {title} ===", flush=True)
    t0 = time.time()
    r = subprocess.run([PY, *args], cwd=ROOT)
    if r.returncode != 0:
        sys.exit(f"Шаг «{title}» завершился с ошибкой (код {r.returncode}). Исправьте и запустите снова — "
                 f"готовые шаги будут пропущены.")
    print(f"--- {title}: {time.time() - t0:.0f} с", flush=True)


def ensure_data():
    dst = ROOT / "data" / "txl-pbc"
    if (dst / "images" / "train").exists():
        return
    print("Скачиваю TXL-PBC …")
    tmp = ROOT / "data" / "_download"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    if shutil.which("git"):
        subprocess.run(["git", "clone", "--depth", "1", DATA_REPO + ".git", str(tmp / "repo")], check=True)
        src = tmp / "repo" / "TXL-PBC"
    else:
        z = tmp / "repo.zip"
        urllib.request.urlretrieve(DATA_REPO + "/archive/refs/heads/master.zip", z)
        zipfile.ZipFile(z).extractall(tmp)
        src = next(tmp.glob("TXL-PBC_Dataset-*")) / "TXL-PBC"
    shutil.copytree(src, dst)
    shutil.rmtree(tmp, ignore_errors=True)


def ensure_weights():
    (ROOT / "weights").mkdir(exist_ok=True)
    for w in WEIGHTS:
        p = ROOT / "weights" / w
        if not p.exists():
            print("Скачиваю", w)
            urllib.request.urlretrieve(f"https://github.com/ultralytics/assets/releases/download/v8.3.0/{w}", p)


def detector_done(name, imgsz, epochs):
    """Готов ли запуск с ТЕМИ ЖЕ настройками. Запуск с другими (например, CPU-версия из архива)
    переносится в runs/previous/, чтобы не перепутать результаты."""
    meta = ROOT / "runs" / "detect" / name / "train_meta.json"
    if not meta.exists():
        return False
    m = json.loads(meta.read_text())
    if m.get("imgsz") == imgsz and m.get("epochs") == epochs:
        return True
    old = ROOT / "runs" / "previous" / f"{name}_{m.get('imgsz')}px_{m.get('epochs')}ep"
    print(f"{name}: найден запуск с другими настройками ({m.get('imgsz')} px, {m.get('epochs')} эпох) → {old}")
    old.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(meta.parent), str(old))
    rep = ROOT / "reports" / "detector" / name
    if rep.exists():
        shutil.move(str(rep), str(old / "reports"))
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", choices=PROFILES, default="gpu")
    ap.add_argument("--epochs", type=int, help="переопределить число эпох детектора")
    ap.add_argument("--imgsz", type=int, help="переопределить размер входа детектора")
    ap.add_argument("--batch", type=int, help="переопределить batch (уменьшите при нехватке видеопамяти)")
    ap.add_argument("--only-detectors", action="store_true", help="только обучение и оценка детекторов")
    ap.add_argument("--force", action="store_true", help="удалить прошлые результаты и обучить заново")
    args = ap.parse_args()
    imgsz, epochs, batch, runs = PROFILES[args.profile]
    imgsz, epochs, batch = args.imgsz or imgsz, args.epochs or epochs, args.batch or batch

    sys.path.insert(0, str(ROOT / "src"))
    from hemascope.env import device
    print(f"Профиль {args.profile}: {imgsz} px, {epochs} эпох, batch {batch}, устройство {device()}")
    if args.profile == "gpu" and device() == "cpu":
        print("ВНИМАНИЕ: CUDA не найдена — PyTorch установлен без поддержки видеокарты (см. README).")
    if args.force:
        for d in ("runs/detect", "runs/classify", "reports"):
            shutil.rmtree(ROOT / d, ignore_errors=True)

    ensure_data()
    ensure_weights()
    if not args.only_detectors:
        step("EDA", ["scripts/eda.py"])
        step("Калибровка проверки качества", ["scripts/calibrate_quality.py"])
        step("Классический baseline", ["scripts/baseline_cv.py"])

    for name, weights, data, seed in runs:
        last = ROOT / "runs" / "detect" / name / "weights" / "last.pt"
        if not detector_done(name, imgsz, epochs):
            cmd = ["scripts/train_detector.py", "--name", name, "--weights", weights, "--data", data,
                   "--epochs", str(epochs), "--imgsz", str(imgsz), "--batch", str(batch), "--seed", str(seed)]
            step(f"Обучение {name}", cmd + (["--resume"] if last.exists() else []))
        metrics = ROOT / "reports" / "detector" / name / "metrics.json"
        meta = ROOT / "runs" / "detect" / name / "train_meta.json"
        if not metrics.exists() or metrics.stat().st_mtime < meta.stat().st_mtime:
            extra = ["--exclude-val-source", "Raabin-WBC"] if data == "loso_raabin" else []
            step(f"Оценка {name}", ["scripts/evaluate_detector.py", "--name", name, *extra])

    if args.only_detectors:
        step("Сводка", ["scripts/analyze.py"])
        return
    step("Вырезка лейкоцитов", ["scripts/make_wbc_crops.py"])
    if not (ROOT / "reports" / "wbc" / "wbc_metrics.json").exists():
        step("Классификатор лейкоцитов", ["scripts/train_wbc.py"])
    if not (ROOT / "reports" / "wbc" / "wbc_cv.json").exists():
        step("Кросс-валидация классификатора", ["scripts/wbc_cv.py"])
    step("Экспорт ONNX", ["scripts/export_onnx.py", "--name", "yolo11n_pretrained"])
    step("Устойчивость к плохим снимкам", ["scripts/robustness.py"])
    step("Углублённый анализ", ["scripts/deep_analysis.py"])
    step("Сводка", ["scripts/analyze.py"])
    step("Тесты", ["-m", "pytest", "-q", "tests"])
    summary = json.loads((ROOT / "reports" / "detector" / "yolo11n_pretrained" / "metrics.json").read_text())
    print("\nГотово. Финальная модель: mAP50 {:.3f}, mAP50-95 {:.3f}, macro F1 {:.3f}".format(
        summary["map_test"]["mAP50"], summary["map_test"]["mAP50_95"], summary["test"]["macro_f1"]))
    print("Отчёт: reports/results/results.md   Демо: python app/app.py")


if __name__ == "__main__":
    main()
