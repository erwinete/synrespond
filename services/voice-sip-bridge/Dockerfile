FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Для Silero VAD раскомментируйте установку pipecat:
# RUN pip install --no-cache-dir "pipecat-ai[silero]>=0.0.60"

COPY config.py vad.py speech.py session.py audiosocket.py server.py ./

EXPOSE 8765

HEALTHCHECK --interval=30s --timeout=5s --retries=5 \
  CMD python -c "import urllib.request;urllib.request.urlopen('http://127.0.0.1:8765/health')" || exit 1

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8765"]
