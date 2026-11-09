"""Юнит-тесты ресемплинга voice-sip-bridge (Д4) и эмоционального прокси (Ф3).

soxr может отсутствовать — тест проходит на fallback-пути и помечает движок.
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

BRIDGE_DIR = Path(__file__).resolve().parents[1] / 'services' / 'voice-sip-bridge'


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# config без внешних зависимостей, speech импортирует config + numpy + aiohttp
_load('config', BRIDGE_DIR / 'config.py')
speech = _load('speech', BRIDGE_DIR / 'speech.py')
emotion = _load('emotion', BRIDGE_DIR / 'emotion.py')


def _tone(freq: float, seconds: float, rate: int) -> bytes:
    t = np.linspace(0, seconds, int(rate * seconds), False)
    samples = (np.sin(2 * np.pi * freq * t) * 12000).astype(np.int16)
    return samples.tobytes()


class TestResample:
    def test_same_rate_is_passthrough(self):
        pcm = _tone(440, 0.1, 8000)
        assert speech.resample(pcm, 8000, 8000) == pcm

    def test_upsample_8k_to_16k_length_doubles(self):
        pcm = _tone(440, 1.0, 8000)
        out = speech.resample(pcm, 8000, 16000)
        in_count = len(pcm) // 2
        out_count = len(out) // 2
        assert abs(out_count - in_count * 2) <= 4

    def test_downsample_24k_to_8k_length_third(self):
        pcm = _tone(300, 1.2, 24000)
        out = speech.resample(pcm, 24000, 8000)
        in_count = len(pcm) // 2
        out_count = len(out) // 2
        assert abs(out_count - in_count / 3) <= 4

    def test_tone_frequency_preserved_after_resample(self):
        # Основная энергия тона 440 Гц должна остаться около 440 Гц после 8к -> 16к
        pcm = _tone(440, 0.5, 8000)
        out = np.frombuffer(speech.resample(pcm, 8000, 16000), dtype=np.int16).astype(np.float64)
        spectrum = np.abs(np.fft.rfft(out))
        peak_freq = np.argmax(spectrum) * 16000 / len(out)
        assert abs(peak_freq - 440) < 25, f'Peak drifted to {peak_freq:.0f} Hz'

    def test_empty_input(self):
        assert speech.resample(b'', 8000, 16000) == b''


class TestEmotionProxy:
    def test_silence_scores_low(self):
        silence = np.zeros(16000, dtype=np.int16).tobytes()
        score = emotion.estimate(silence, 16000)
        assert score is not None and score < 0.3

    def test_loud_signal_scores_higher_than_silence(self):
        silence = np.zeros(16000, dtype=np.int16).tobytes()
        shout = (_tone(200, 1.0, 16000) * 1)  # амплитуда 12000
        loud_score = emotion.estimate(shout, 16000)
        quiet_score = emotion.estimate(silence, 16000)
        assert loud_score > quiet_score

    def test_score_bounded(self):
        shout = _tone(200, 1.0, 16000)
        score = emotion.estimate(shout, 16000)
        assert 0.0 <= score <= 1.0

    def test_empty_is_none(self):
        assert emotion.estimate(b'', 8000) is None


class TestWavParsing:
    def test_wav_to_pcm_extracts_data_chunk(self):
        import struct

        pcm = _tone(440, 0.05, 8000)
        header = b'RIFF' + struct.pack('<I', 36 + len(pcm)) + b'WAVE'
        header += b'fmt ' + struct.pack('<IHHIIHH', 16, 1, 1, 8000, 16000, 2, 16)
        header += b'data' + struct.pack('<I', len(pcm))
        assert speech.wav_to_pcm(header + pcm) == pcm
