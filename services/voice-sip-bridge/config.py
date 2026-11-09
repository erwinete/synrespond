"""Конфигурация voice-sip-bridge из переменных окружения."""

import os

# Open WebUI: полный RAG-пайплайн бота
OPEN_WEBUI_URL = os.getenv('OPEN_WEBUI_URL', 'http://open-webui:8080').rstrip('/')
OPEN_WEBUI_TOKEN = os.getenv('OPEN_WEBUI_TOKEN', '')
DEFAULT_CHAT_MODEL = os.getenv('VOICE_CHAT_MODEL', 'support-voice-bot')

# Speaches: OpenAI-совместимый STT/TTS
SPEACHES_URL = os.getenv('SPEACHES_URL', 'http://speaches:8000/v1').rstrip('/')
STT_MODEL = os.getenv('VOICE_STT_MODEL', 'faster-whisper-large-v3-turbo')
TTS_MODEL = os.getenv('VOICE_TTS_MODEL', 'kokoro')
TTS_VOICE = os.getenv('VOICE_TTS_VOICE', 'af_heart')
TTS_RESPONSE_FORMAT = os.getenv('VOICE_TTS_FORMAT', 'wav')
STT_LANGUAGE = os.getenv('VOICE_STT_LANGUAGE') or None

# Транспорт
AUDIOSOCKET_HOST = os.getenv('AUDIOSOCKET_HOST', '0.0.0.0')
AUDIOSOCKET_PORT = int(os.getenv('AUDIOSOCKET_PORT', '8765'))

# Аудио: Asterisk AudioSocket отдает slin 8 кГц; whisper/kokoro работают на 16/24 кГц
TELEPHONY_SAMPLE_RATE = 8000
INTERNAL_SAMPLE_RATE = 16000

# VAD (мс): старт речи, конец реплики, максимум сегмента, префикс перед стартом
VAD_START_MS = int(os.getenv('VAD_START_MS', '120'))
VAD_END_MS = int(os.getenv('VAD_END_MS', '600'))
VAD_MAX_SEGMENT_MS = int(os.getenv('VAD_MAX_SEGMENT_MS', '15000'))
VAD_PREFIX_MS = int(os.getenv('VAD_PREFIX_MS', '300'))

# Barge-in: прерывать воспроизведение TTS при новой речи абонента
BARGE_IN_ENABLED = os.getenv('VOICE_BARGE_IN', 'True').lower() == 'true'

# Д3: акустический филлер маскирует задержку тяжелого RAG в телефонии
FILLER_ENABLED = os.getenv('VOICE_FILLER', 'True').lower() == 'true'
FILLER_DELAY_MS = int(os.getenv('VOICE_FILLER_DELAY_MS', '250'))
FILLER_TEXT = os.getenv('VOICE_FILLER_TEXT', 'Секунду, проверяю информацию.')

# Д4 (Stage 5): локальный голосовой контур без speaches
FALLBACK_ON_EMPTY_STT = os.getenv('VOICE_FALLBACK_ON_EMPTY', 'True').lower() == 'true'
FALLBACK_TEXT = os.getenv('VOICE_FALLBACK_TEXT', 'Здравствуйте! Я вас слышу. Задайте ваш вопрос.')

# Д5 (Stage 3): SIP-эскалация — очередь Asterisk для перевода звонка
HANDOFF_QUEUE = os.getenv('VOICE_HANDOFF_QUEUE', 'support_queue')
HANDOFF_ANGER_THRESHOLD = float(os.getenv('VOICE_HANDOFF_ANGER', '0.75'))
REDIS_URL = os.getenv('REDIS_URL', '')

# Ф3: эмоции абонента (гнев/срочность). ONNX-модель опциональна, иначе акустические прокси
EMOTION_ENABLED = os.getenv('VOICE_EMOTION', 'True').lower() == 'true'
EMOTION_MODEL_PATH = os.getenv('VOICE_EMOTION_MODEL_PATH', '')
EMOTION_SAMPLE_WINDOW_S = float(os.getenv('VOICE_EMOTION_WINDOW_S', '2.0'))

# Гистерезис энергетического VAD (если pipecat недоступен)
ENERGY_THRESHOLD = float(os.getenv('VAD_ENERGY_THRESHOLD', '500.0'))
