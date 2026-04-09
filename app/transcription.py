import logging
import time

from faster_whisper import WhisperModel
from app.config import settings

logger = logging.getLogger(__name__)

_whisper_model = None


def get_whisper_model():
    global _whisper_model

    if _whisper_model is None:
        started = time.perf_counter()
        logger.info(
            "Initializing WhisperModel size=%s compute_type=%s",
            settings.whisper_model_size,
            settings.whisper_compute_type,
        )
        _whisper_model = WhisperModel(
            settings.whisper_model_size,
            device="cpu",
            compute_type=settings.whisper_compute_type,
        )
        logger.info(
            "WhisperModel initialized successfully init_sec=%.3f",
            time.perf_counter() - started,
        )

    return _whisper_model


def transcribe_audio(file_path: str):
    started = time.perf_counter()
    logger.info("transcribe_audio called file_path=%s", file_path)

    try:
        model = get_whisper_model()

        logger.info("Whisper transcribe started file_path=%s", file_path)
        segments, info = model.transcribe(
            file_path,
            task="transcribe",
            language="ru",                  # сначала тестируй ru
            beam_size=5,
            vad_filter=True,
            condition_on_previous_text=False,
        )

        logger.info(
            "Whisper returned iterator language=%s duration=%s",
            getattr(info, "language", None),
            getattr(info, "duration", None),
        )

        text_parts = []
        segment_count = 0

        for segment in segments:
            segment_count += 1
            segment_text = segment.text.strip()
            text_parts.append(segment_text)

            if segment_count <= 5 or segment_count % 20 == 0:
                logger.info(
                    "Segment parsed idx=%s start=%.2f end=%.2f text_len=%s",
                    segment_count,
                    getattr(segment, "start", 0.0),
                    getattr(segment, "end", 0.0),
                    len(segment_text),
                )

        transcript = " ".join(part for part in text_parts if part).strip()

        logger.info(
            "Transcription completed file_path=%s segments=%s text_len=%s total_sec=%.3f",
            file_path,
            segment_count,
            len(transcript),
            time.perf_counter() - started,
        )

        return {
            "language": getattr(info, "language", None),
            "duration": getattr(info, "duration", None),
            "text": transcript,
        }

    except Exception:
        logger.exception("transcribe_audio failed file_path=%s", file_path)
        raise