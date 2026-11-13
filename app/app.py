"""HemaScope — демо-приложение.

Запуск:  python app/app.py                 (PyTorch-веса)
         python app/app.py --onnx          (ускоренная ONNX-версия детектора)
"""
import argparse
import sys
from pathlib import Path

import gradio as gr
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from hemascope.data import CLASS_NAMES  # noqa: E402
from hemascope.pipeline import HemaScope  # noqa: E402

RU = {"WBC": "Лейкоциты", "RBC": "Эритроциты", "Platelets": "Тромбоциты"}
DISCLAIMER = (
    "**Исследовательский прототип, не медицинское изделие.** Считает клетки в поле зрения; "
    "концентрацию (клеток/мкл) без калибровки микроскопа не определяет. Классификатор лейкоцитов "
    "обучен на 5 типах из одного набора (PBC): сегментоядерных нейтрофилов в нём нет, поэтому они "
    "будут отнесены к ближайшему типу. Результат проверяет лаборант."
)


def build(model: HemaScope):
    def one_field(img):
        if img is None:
            return None, "Загрузите снимок", pd.DataFrame(), pd.DataFrame()
        r = model.analyze_field(Image.fromarray(img))
        if not r.quality.ok:
            status = "### ⚠ Переснимите поле зрения\n" + "\n".join(f"- {p}" for p in r.quality.problems)
        else:
            status = "### ✓ Качество снимка в норме"
        status += f"\n\nРезкость {r.quality.sharpness:.3f}, яркость {r.quality.brightness:.0f}"
        counts = pd.DataFrame([{"Клетки": RU[n], "Найдено": v} for n, v in r.counts().items()])
        wbc = pd.DataFrame([{"Тип": t["type_ru"], "Уверенность": f"{t['confidence']:.0%}",
                             "Статус": "проверить вручную" if t["uncertain"] else "ок"} for t in r.wbc_types])
        return r.annotated, status, counts, wbc

    def smear(files):
        if not files:
            return "Загрузите несколько снимков одного мазка", pd.DataFrame(), pd.DataFrame(), []
        paths = [f if isinstance(f, str) else f.name for f in files]
        s = model.analyze_smear(paths)
        rej = s["fields_total"] - s["fields_used"]
        md = (f"### Полей зрения: {s['fields_total']}, использовано {s['fields_used']}"
              + (f" (отклонено по качеству: {rej})" if rej else "")
              + f"\n\nЭритроцитов на один лейкоцит: **{s['rbc_per_wbc']}**"
              + f"\n\nЛейкоцитов классифицировано: {s['wbc_classified']}, неуверенных: {s['wbc_uncertain']}"
              + ("\n\n> Для надёжной лейкоцитарной формулы лаборант считает 100 лейкоцитов; "
                 "здесь их меньше — формула ориентировочная." if s["wbc_classified"] < 100 else ""))
        counts = pd.DataFrame([{"Клетки": RU[n], "Всего": s["counts"][n], "В среднем на поле": s["per_field_mean"][n]}
                               for n in CLASS_NAMES])
        formula = pd.DataFrame([{"Тип": k, "%": v} for k, v in s["leukocyte_formula_pct"].items()])
        gallery = [(f.annotated, f"поле {i + 1}" + ("" if f.quality.ok else " — отклонено"))
                   for i, f in enumerate(s["fields"])]
        return md, counts, formula, gallery

    examples = sorted(str(p) for p in (ROOT / "app" / "examples").glob("*.png"))
    with gr.Blocks(title="HemaScope") as demo:
        gr.Markdown("# HemaScope — AI-подсчёт клеток крови в мазке")
        gr.Markdown(DISCLAIMER)
        with gr.Tab("Одно поле зрения"):
            with gr.Row():
                with gr.Column():
                    inp = gr.Image(label="Снимок мазка (микроскоп, ×100 иммерсия)", type="numpy")
                    btn = gr.Button("Анализировать", variant="primary")
                    if examples:
                        gr.Examples(examples, inputs=inp, label="Примеры из тестовой выборки")
                with gr.Column():
                    out_img = gr.Image(label="Найденные клетки: фиолетовые — лейкоциты, красные — эритроциты, "
                                             "бирюзовые — тромбоциты")
                    status = gr.Markdown()
                    counts = gr.Dataframe(label="Подсчёт")
                    wbc = gr.Dataframe(label="Типы лейкоцитов")
            btn.click(one_field, inp, [out_img, status, counts, wbc])
        with gr.Tab("Мазок целиком (несколько полей)"):
            files = gr.File(file_count="multiple", file_types=["image"], label="Снимки полей зрения одного мазка")
            btn2 = gr.Button("Посчитать по мазку", variant="primary")
            md = gr.Markdown()
            with gr.Row():
                c2 = gr.Dataframe(label="Подсчёт по всем полям")
                f2 = gr.Dataframe(label="Лейкоцитарная формула, %")
            gal = gr.Gallery(label="Поля зрения", columns=4, height=360)
            btn2.click(smear, files, [md, c2, f2, gal])
    return demo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--detector", default="yolo11n_pretrained")
    ap.add_argument("--onnx", action="store_true")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--share", action="store_true")
    args = ap.parse_args()
    model = HemaScope.from_run(args.detector, onnx=args.onnx)
    build(model).launch(server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
