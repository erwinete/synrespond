"""Экспорт HF-модели в ONNX для каталога классификатора.

Примеры:
  # Классификатор интентов (sequence classification)
  python scripts/export_onnx.py --model ai-forever/ruBert-base \
      --task sequence_classification --out ../../models/encoders/rubert-intents-v1

  # NLI для zero-shot
  python scripts/export_onnx.py --model MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7 \
      --task zero_shot_nli --entailment-id 2 --out ../../models/encoders/deberta-v3-nli

  # NER / слоты
  python scripts/export_onnx.py --model Davlan/bert-base-cased-ner-hrl \
      --task token_classification --out ../../models/encoders/slots-ner-v1

Требования: pip install torch transformers onnx optimum onnxruntime
"""

import argparse
import json
import os
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Экспорт HF-модели в ONNX")
    parser.add_argument("--model", required=True, help="HF model id или локальный путь")
    parser.add_argument("--out", required=True, help="Каталог вывода MODELS_DIR/<model_id>")
    parser.add_argument(
        "--task",
        default="sequence_classification",
        choices=["sequence_classification", "zero_shot_nli", "token_classification", "feature_extraction"],
    )
    parser.add_argument(
        "--init-new-head",
        action="store_true",
        help="Создать новую классификационную голову (ignore_mismatched_sizes) перед экспортом",
    )
    parser.add_argument(
        "--zero-shot-mode",
        default=None,
        choices=[None, "embedding"],
        help="Режим zero-shot для label_map.json",
    )
    parser.add_argument("--labels", help="JSON-файл или список меток через запятую (для классификации)")
    parser.add_argument("--hypothesis-template", default="This example is {}.")
    parser.add_argument("--entailment-id", type=int, default=2, help="Индекс колонки entailment в NLI-логитах")
    parser.add_argument("--tau-high", type=float, default=0.85)
    parser.add_argument("--tau-low", type=float, default=0.50)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--quantize", action="store_true", help="INT8-квантование (dynamic)")
    return parser.parse_args()


def resolve_labels(args: argparse.Namespace, num_classes: int | None) -> dict[int, str]:
    if args.labels:
        if os.path.isfile(args.labels):
            raw = json.loads(Path(args.labels).read_text(encoding="utf-8"))
            return {int(k): str(v) for k, v in raw.items()}
        names = [s.strip() for s in args.labels.split(",") if s.strip()]
        return dict(enumerate(names))
    return {i: f"class_{i}" for i in range(num_classes or 0)}


def main() -> None:
    args = parse_args()
    from transformers import AutoModelForSequenceClassification, AutoModelForTokenClassification, AutoTokenizer
    from optimum.exporters.onnx import main_export

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # BUG-09: токенизатор загружается до блока --init-new-head и доступен
    # для сохранения во временную папку модели с новой головой
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    task_map = {
        "sequence_classification": "text-classification",
        "zero_shot_nli": "text-classification",
        "token_classification": "token-classification",
        "feature_extraction": "feature-extraction",
    }
    model_cls = AutoModelForTokenClassification if args.task == "token_classification" else AutoModelForSequenceClassification

    export_source = args.model
    tmp_model_dir = None
    if args.init_new_head and args.task in ("sequence_classification", "zero_shot_nli", "token_classification"):
        # Новая голова под переданные метки (для экспорта без совпадения размеров)
        import shutil
        import tempfile

        num_labels = len(resolve_labels(args, None)) or 2
        model = model_cls.from_pretrained(args.model, num_labels=num_labels, ignore_mismatched_sizes=True)
        tmp_model_dir = tempfile.mkdtemp(prefix="hf_head_")
        try:
            model.save_pretrained(tmp_model_dir)
            tokenizer.save_pretrained(tmp_model_dir)
            export_source = tmp_model_dir
            print(f"Создана новая голова на {num_labels} классов -> экспорт из {tmp_model_dir}")

            print(f"Загрузка {args.model} и экспорт в ONNX ({task_map[args.task]})...")
            main_export(
                export_source,
                output=str(out_dir),
                task=task_map[args.task],
                opset=17,
                device="cpu",
            )
        finally:
            # BUG-09: временная папка не протекает при ошибке экспорта
            shutil.rmtree(tmp_model_dir, ignore_errors=True)
    else:
        print(f"Загрузка {args.model} и экспорт в ONNX ({task_map[args.task]})...")
        main_export(
            export_source,
            output=str(out_dir),
            task=task_map[args.task],
            opset=17,
            device="cpu",
        )

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.save_pretrained(out_dir)

    # Число классов — из конфигурации экспортированной модели
    num_classes = None
    config_path = out_dir / "config.json"
    if config_path.exists():
        cfg = json.loads(config_path.read_text(encoding="utf-8"))
        num_classes = cfg.get("num_labels") or len(cfg.get("id2label", {})) or None

    labels = {} if args.task == "feature_extraction" else resolve_labels(args, num_classes)
    meta = {
        "task": args.task,
        "labels": labels,
        "max_length": args.max_length,
        "thresholds": {"high": args.tau_high, "low": args.tau_low},
        "source_model": args.model,
    }
    if args.zero_shot_mode:
        meta["zero_shot_mode"] = args.zero_shot_mode
    if args.task == "zero_shot_nli":
        meta["hypothesis_template"] = args.hypothesis_template
        meta["entailment_id"] = args.entailment_id
    if args.task == "token_classification":
        meta["aggregation_threshold"] = 0.6
        meta["labels"] = {int(k): v for k, v in (num_classes and {i: f"label_{i}" for i in range(num_classes)}.items() or labels.items())}

    (out_dir / "label_map.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.quantize:
        from onnxruntime.quantization import quantize_dynamic

        model_path = out_dir / "model.onnx"
        quantized = out_dir / "model.int8.onnx"
        quantize_dynamic(str(model_path), str(quantized))
        model_path.unlink()
        quantized.rename(model_path)
        print("Применено INT8-квантование.")

    print(f"Готово: {out_dir}")


if __name__ == "__main__":
    main()
