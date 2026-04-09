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
    else:
        logger.info("WhisperModel already initialized, reusing existing instance")

    return _whisper_model


def _run_transcription(
    model,
    file_path: str,
    mode_name: str,
    language: str | None = None,
    vad_filter: bool = False,
    multilingual: bool = False,
):
    logger.info(
        "Whisper run started file_path=%s mode=%s language=%s vad_filter=%s multilingual=%s",
        file_path,
        mode_name,
        language,
        vad_filter,
        multilingual,
    )

    started = time.perf_counter()

    segments, info = model.transcribe(
        file_path,
        task="transcribe",
        language=language,
        beam_size=5,
        vad_filter=vad_filter,
        multilingual=multilingual,
        condition_on_previous_text=False,
        word_timestamps=True,
    )

    logger.info(
        "Whisper returned iterator mode=%s detected_language=%s duration=%s first_stage_sec=%.3f",
        mode_name,
        getattr(info, "language", None),
        getattr(info, "duration", None),
        time.perf_counter() - started,
    )

    text_parts = []
    segment_count = 0

    for segment in segments:
        segment_count += 1
        segment_text = segment.text.strip()
        text_parts.append(segment_text)

        if segment_count <= 5 or segment_count % 20 == 0:
            logger.info(
                "Segment parsed mode=%s idx=%s start=%.2f end=%.2f text_len=%s",
                mode_name,
                segment_count,
                getattr(segment, "start", 0.0),
                getattr(segment, "end", 0.0),
                len(segment_text),
            )

    transcript = " ".join(part for part in text_parts if part).strip()

    logger.info(
        "Whisper run completed mode=%s segments=%s text_len=%s total_sec=%.3f",
        mode_name,
        segment_count,
        len(transcript),
        time.perf_counter() - started,
    )

    return {
        "language": getattr(info, "language", None),
        "duration": getattr(info, "duration", None),
        "text": transcript,
        "segment_count": segment_count,
        "mode": mode_name,
    }


def _score_result(result: dict) -> int:
    text = (result.get("text") or "").strip()
    segment_count = result.get("segment_count", 0)
    language = result.get("language")

    score = 0

    score += len(text)
    score += segment_count * 10

    if language in {"ru", "kk"}:
        score += 50

    # штраф за подозрительное повторение
    words = text.lower().split()
    if len(words) >= 6:
        uniq = len(set(words))
        if uniq / max(len(words), 1) < 0.45:
            score -= 80

    return score


def transcribe_audio(file_path: str):
    started = time.perf_counter()
    logger.info("transcribe_audio called file_path=%s", file_path)

    try:
        model = get_whisper_model()
        candidates = []

        # 1. RU без VAD — чтобы не съедать начало короткого звонка
        candidates.append(
            _run_transcription(
                model=model,
                file_path=file_path,
                mode_name="ru_no_vad",
                language="ru",
                vad_filter=False,
                multilingual=False,
            )
        )

        # 2. RU с VAD
        candidates.append(
            _run_transcription(
                model=model,
                file_path=file_path,
                mode_name="ru_vad",
                language="ru",
                vad_filter=True,
                multilingual=False,
            )
        )

        # 3. Multilingual без фиксированного языка
        candidates.append(
            _run_transcription(
                model=model,
                file_path=file_path,
                mode_name="multi_no_vad",
                language=None,
                vad_filter=False,
                multilingual=True,
            )
        )

        for c in candidates:
            logger.info(
                "Candidate mode=%s detected_language=%s text_len=%s segment_count=%s score=%s",
                c.get("mode"),
                c.get("language"),
                len(c.get("text", "")),
                c.get("segment_count"),
                _score_result(c),
            )

        final_result = max(candidates, key=_score_result)

        logger.info(
            "Selected transcription final_mode=%s final_language=%s text_len=%s total_sec=%.3f",
            final_result.get("mode"),
            final_result.get("language"),
            len(final_result.get("text", "")),
            time.perf_counter() - started,
        )

        return {
            "language": final_result.get("language"),
            "duration": final_result.get("duration"),
            "text": final_result.get("text"),
            "mode": final_result.get("mode"),
        }

    except Exception:
        logger.exception("transcribe_audio failed file_path=%s", file_path)
        raise