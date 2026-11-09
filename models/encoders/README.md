Каталог ONNX-моделей для encoder-classifier (монтируется в контейнер как /models).

Структура каждой модели:
  <model_id>/model.onnx, tokenizer.json, label_map.json

Экспорт: python ../../services/encoder-classifier/scripts/export_onnx.py --model <hf_id> --out <model_id>
