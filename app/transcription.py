import logging
import math
import os
import subprocess
import tempfile
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


def get_audio_duration(file_path: str) -> float:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        file_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return float(result.stdout.strip())


def split_audio_to_chunks(file_path: str, chunk_sec: int = 4) -> list[tuple[str, float]]:
    duration = get_audio_duration(file_path)
    logger.info("Audio duration for chunking: %.2f sec", duration)

    chunks = []
    tmp_dir = tempfile.mkdtemp(prefix="stt_chunks_")

    total_chunks = math.ceil(duration / chunk_sec)

    for i in range(total_chunks):
        start = i * chunk_sec
        out_path = os.path.join(tmp_dir, f"chunk_{i:03d}.wav")

        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            file_path,
            "-ss",
            str(start),
            "-t",
            str(chunk_sec),
            "-ac",
            "1",
            "-ar",
            "16000",
            out_path,
        ]

        subprocess.run(cmd, capture_output=True, text=True, check=True)
        chunks.append((out_path, float(start)))

    logger.info("Audio split into %s chunks", len(chunks))
    return chunks


def cleanup_chunks(chunks: list[tuple[str, float]]):
    for chunk_path, _ in chunks:
        try:
            if os.path.exists(chunk_path):
                os.remove(chunk_path)
        except Exception:
            logger.exception("Failed to delete chunk %s", chunk_path)

    if chunks:
        chunk_dir = os.path.dirname(chunks[0][0])
        try:
            if os.path.isdir(chunk_dir):
                os.rmdir(chunk_dir)
        except Exception:
            logger.exception("Failed to delete chunk dir %s", chunk_dir)


def _run_transcription(model, file_path: str, mode_name: str, language: str | None = None):
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

    score = 0
    score += len(text)
    score += segment_count * 10

    if language in {"ru", "kk"}:
        score += 50

    words = text.lower().split()
    if len(words) >= 6:
        uniq_ratio = len(set(words)) / max(len(words), 1)
        if uniq_ratio < 0.45:
            score -= 80

    return score


def _transcribe_chunk(model, chunk_path: str, chunk_start: float) -> dict:
    candidates = []

    candidates.append(_run_transcription(model, chunk_path, "kk", "kk"))
    candidates.append(_run_transcription(model, chunk_path, "ru", "ru"))
    candidates.append(_run_transcription(model, chunk_path, "auto", None))

    for c in candidates:
        logger.info(
            "Chunk start=%.2f candidate mode=%s detected_language=%s text_len=%s segment_count=%s score=%s",
            chunk_start,
            c.get("mode"),
            c.get("language"),
            len(c.get("text", "")),
            c.get("segment_count"),
            _score_result(c),
        )

    best = max(candidates, key=_score_result)

    logger.info(
        "Chunk start=%.2f selected mode=%s detected_language=%s text_len=%s",
        chunk_start,
        best.get("mode"),
        best.get("language"),
        len(best.get("text", "")),
    )

    return best


def transcribe_audio(file_path: str):
    started = time.perf_counter()
    logger.info("transcribe_audio called file_path=%s", file_path)

    chunks = []
    try:
        model = get_whisper_model()

        chunks = split_audio_to_chunks(file_path, chunk_sec=4)

        final_parts = []
        detected_languages = []

        for chunk_path, chunk_start in chunks:
            best = _transcribe_chunk(model, chunk_path, chunk_start)

            text = (best.get("text") or "").strip()
            if text:
                final_parts.append(text)

            lang = best.get("language")
            if lang:
                detected_languages.append(lang)

        final_text = " ".join(final_parts).strip()

        if detected_languages:
            final_language = max(set(detected_languages), key=detected_languages.count)
        else:
            final_language = None

        logger.info(
            "Transcription final completed language=%s text_len=%s total_sec=%.3f",
            final_language,
            len(final_text),
            time.perf_counter() - started,
        )

        return {
            "language": final_language,
            "duration": get_audio_duration(file_path),
            "text": final_text,
            "mode": "chunked_multi_pass",
        }

    except Exception:
        logger.exception("transcribe_audio failed file_path=%s", file_path)
        raise
    finally:
        if chunks:
            cleanup_chunks(chunks)