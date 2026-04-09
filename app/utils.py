import logging
import os
import subprocess
import time
import uuid
from pathlib import Path

from fastapi import UploadFile

from app.config import settings


logger = logging.getLogger(__name__)


def ensure_upload_dir():
    started = time.perf_counter()
    Path(settings.upload_dir).mkdir(parents=True, exist_ok=True)
    logger.info(
        "Upload dir ensured path=%s ensure_sec=%.3f",
        settings.upload_dir,
        time.perf_counter() - started,
    )


async def save_upload_file(upload_file: UploadFile) -> str:
    started = time.perf_counter()

    logger.info(
        "save_upload_file called filename=%s content_type=%s",
        upload_file.filename,
        upload_file.content_type,
    )

    ensure_upload_dir()

    ext = ""
    if upload_file.filename and "." in upload_file.filename:
        ext = "." + upload_file.filename.split(".")[-1].lower()

    file_name = f"{uuid.uuid4().hex}{ext}"
    file_path = os.path.join(settings.upload_dir, file_name)

    logger.info("Temp file path generated path=%s", file_path)

    read_started = time.perf_counter()
    content = await upload_file.read()
    logger.info(
        "Upload file fully read size_bytes=%s read_sec=%.3f",
        len(content),
        time.perf_counter() - read_started,
    )

    write_started = time.perf_counter()
    with open(file_path, "wb") as f:
        f.write(content)

    logger.info(
        "Upload file written path=%s size_bytes=%s write_sec=%.3f total_sec=%.3f",
        file_path,
        len(content),
        time.perf_counter() - write_started,
        time.perf_counter() - started,
    )

    return file_path


def preprocess_audio_ffmpeg(file_path: str) -> str:
    """
    Мягкий preprocessing:
    - mono
    - 16kHz
    Без агрессивной фильтрации по умолчанию.
    """
    started = time.perf_counter()

    base, _ = os.path.splitext(file_path)
    output_path = f"{base}_preprocessed.wav"

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        file_path,
        "-ac",
        "1",
        "-ar",
        "16000",
        output_path,
    ]

    logger.info("Starting ffmpeg preprocess input=%s output=%s", file_path, output_path)

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
        )

        logger.info(
            "ffmpeg preprocess completed output=%s sec=%.3f",
            output_path,
            time.perf_counter() - started,
        )

        if result.stderr:
            logger.info("ffmpeg stderr: %s", result.stderr[-1200:])

        return output_path

    except subprocess.CalledProcessError:
        logger.exception("ffmpeg preprocess failed input=%s", file_path)
        raise


def delete_file_safely(file_path: str):
    started = time.perf_counter()

    try:
        if not file_path:
            logger.warning("delete_file_safely called with empty path")
            return

        if os.path.exists(file_path):
            os.remove(file_path)
            logger.info(
                "Temp file deleted path=%s delete_sec=%.3f",
                file_path,
                time.perf_counter() - started,
            )
        else:
            logger.warning("Temp file does not exist, skip delete path=%s", file_path)

    except Exception:
        logger.exception("Failed to delete temp file path=%s", file_path)