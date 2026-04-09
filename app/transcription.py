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
):
    logger.info(
        "Whisper run started file_path=%s mode=%s language=%s",
        file_path,
        mode_name,
        language,
    )

    started = time.perf_counter()

    segments, info = model.transcribe(
        file_path,
        task="transcribe",
        language=language,
        beam_size=5,
        vad_filter=False,
        condition_on_previous_text=False,
        word_timestamps=False,
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
        if segment_text:
            text_parts.append(segment_text)

    transcript = " ".join(text_parts).strip()

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

    if not text:
        return -1000

    score = 0
    score += min(len(text), 250)
    score += segment_count * 12

    if language in {"ru", "kk"}:
        score += 50
    else:
        score -= 20

    words = text.lower().split()
    word_count = len(words)

    if word_count <= 1:
        score -= 120
    elif word_count <= 3:
        score -= 50

    if word_count >= 4:
        uniq_ratio = len(set(words)) / max(word_count, 1)
        if uniq_ratio < 0.40:
            score -= 120
        elif uniq_ratio < 0.55:
            score -= 60

    if segment_count == 1 and len(text) > 60:
        score -= 25

    cyrillic_count = sum(
        1
        for ch in text
        if ("А" <= ch <= "я") or ch in "ӘәІіҢңҒғҮүҰұҚқӨөҺһЁё"
    )
    if cyrillic_count > 0:
        score += 20

    return score


def transcribe_audio(file_path: str):
    started = time.perf_counter()
    logger.info("transcribe_audio called file_path=%s", file_path)

    try:
        model = get_whisper_model()

        result_kk = _run_transcription(
            model=model,
            file_path=file_path,
            mode_name="kk_whole",
            language="kk",
        )

        result_ru = _run_transcription(
            model=model,
            file_path=file_path,
            mode_name="ru_whole",
            language="ru",
        )

        result_auto = _run_transcription(
            model=model,
            file_path=file_path,
            mode_name="auto_whole",
            language=None,
        )

        candidates = [result_kk, result_ru, result_auto]

        for c in candidates:
            logger.info(
                "Candidate mode=%s detected_language=%s text_len=%s segment_count=%s score=%s",
                c.get("mode"),
                c.get("language"),
                len(c.get("text", "")),
                c.get("segment_count"),
                _score_result(c),
            )

        best = max(candidates, key=_score_result)

        logger.info(
            "Selected transcription final_mode=%s final_language=%s text_len=%s total_sec=%.3f",
            best.get("mode"),
            best.get("language"),
            len(best.get("text", "")),
            time.perf_counter() - started,
        )

        return {
            "language": best.get("language"),
            "duration": best.get("duration"),
            "text": best.get("text"),
            "mode": "whole_file_multi_pass",
            "best_mode": best.get("mode"),
            "alternatives": {
                "kk": result_kk.get("text"),
                "ru": result_ru.get("text"),
                "auto": result_auto.get("text"),
            },
        }

    except Exception:
        logger.exception("transcribe_audio failed file_path=%s", file_path)
        raise