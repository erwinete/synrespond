# Сервис инференса энкодеров. База — slim-образ без CUDA.
# Для GPU замените onnxruntime на onnxruntime-gpu и возьмите базу с CUDA
# (см. аргумент BASE_IMAGE и extras-вариант в docker-compose).
ARG BASE_IMAGE=python:3.12-slim
FROM ${BASE_IMAGE}

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MODELS_DIR=/models \
    ONNX_PROVIDERS=auto

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py engine.py decision.py schemas.py ./

# Каталог моделей монтируется как volume
VOLUME ["/models"]
EXPOSE 8001

HEALTHCHECK --interval=30s --timeout=5s --retries=5 \
  CMD python -c "import urllib.request;urllib.request.urlopen('http://127.0.0.1:8001/health')" || exit 1

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8001"]
