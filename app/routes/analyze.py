import json
import logging
import os
import time
import uuid
from json import JSONDecodeError

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.config import settings
from app.ollama_client import generate_with_ollama
from app.prompts import build_analysis_prompt
from app.transcription import transcribe_audio
from app.utils import delete_file_safely, preprocess_audio_ffmpeg, save_upload_file


router = APIRouter(tags=["analysis"])
logger = logging.getLogger(__name__)


@router.get("/health")
async def health():
    logger.info("Health check called")
    return {
        "status": "ok",
        "app": settings.app_name,
        "model": settings.llm_model_name,
    }


@router.post("/transcribe")
async def transcribe_endpoint(file: UploadFile = File(...)):
    request_id = str(uuid.uuid4())[:8]
    started_at = time.perf_counter()
    file_path = None
    preprocessed_path = None

    logger.info(f"[{request_id}] /transcribe request received")

    try:
        logger.info(
            f"[{request_id}] incoming file: filename={file.filename}, content_type={file.content_type}"
        )

        save_started = time.perf_counter()
        file_path = await save_upload_file(file)
        logger.info(
            f"[{request_id}] file saved to temp path={file_path}, save_sec={time.perf_counter() - save_started:.3f}"
        )

        if file_path:
            try:
                file_size = os.path.getsize(file_path)
                logger.info(f"[{request_id}] saved file size_bytes={file_size}")
            except Exception:
                logger.exception(f"[{request_id}] failed to get temp file size")

        logger.info(f"[{request_id}] ffmpeg preprocessing started")
        preprocessed_path = preprocess_audio_ffmpeg(file_path)
        logger.info(f"[{request_id}] ffmpeg preprocessing finished path={preprocessed_path}")

        logger.info(f"[{request_id}] transcription started")
        stt_started = time.perf_counter()

        transcription = transcribe_audio(preprocessed_path)

        logger.info(
            f"[{request_id}] transcription finished stt_sec={time.perf_counter() - stt_started:.3f} "
            f"text_len={len(transcription.get('text', ''))} language={transcription.get('language')} "
            f"best_mode={transcription.get('best_mode')}"
        )

        logger.info(
            f"[{request_id}] /transcribe completed total_sec={time.perf_counter() - started_at:.3f}"
        )

        return {
            "status": "ok",
            "transcription": transcription,
        }

    except Exception as exc:
        logger.exception(f"[{request_id}] /transcribe failed: {exc}")
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(exc)}")

    finally:
        if preprocessed_path:
            logger.info(f"[{request_id}] deleting preprocessed file path={preprocessed_path}")
            delete_file_safely(preprocessed_path)

        if file_path:
            logger.info(f"[{request_id}] deleting temp file path={file_path}")
            delete_file_safely(file_path)


@router.post("/analyze-audio")
async def analyze_audio(file: UploadFile = File(...)):
    request_id = str(uuid.uuid4())[:8]
    started_at = time.perf_counter()
    file_path = None
    preprocessed_path = None

    logger.info(f"[{request_id}] /analyze-audio request received")

    try:
        logger.info(
            f"[{request_id}] incoming file: filename={file.filename}, content_type={file.content_type}"
        )

        save_started = time.perf_counter()
        file_path = await save_upload_file(file)
        logger.info(
            f"[{request_id}] file saved to temp path={file_path}, save_sec={time.perf_counter() - save_started:.3f}"
        )

        if file_path:
            try:
                file_size = os.path.getsize(file_path)
                logger.info(f"[{request_id}] saved file size_bytes={file_size}")
            except Exception:
                logger.exception(f"[{request_id}] failed to get temp file size")

        logger.info(f"[{request_id}] ffmpeg preprocessing started")
        preprocessed_path = preprocess_audio_ffmpeg(file_path)
        logger.info(f"[{request_id}] ffmpeg preprocessing finished path={preprocessed_path}")

        logger.info(f"[{request_id}] transcription started")
        stt_started = time.perf_counter()
        transcription = transcribe_audio(preprocessed_path)

        logger.info(
            f"[{request_id}] transcription finished stt_sec={time.perf_counter() - stt_started:.3f} "
            f"text_len={len(transcription.get('text', ''))} language={transcription.get('language')} "
            f"best_mode={transcription.get('best_mode')}"
        )

        best_text = transcription.get("text", "").strip()
        alternatives = transcription.get("alternatives", {})
        transcript_language = transcription.get("language")

        if not best_text and not any((alternatives or {}).values()):
            logger.warning(f"[{request_id}] empty transcription result")
            raise HTTPException(status_code=400, detail="Empty transcription result")

        prompt = build_analysis_prompt(
            best_text=best_text,
            alternatives=alternatives,
            detected_language=transcript_language,
            conversation_type="crm_customer_call",
        )

        logger.info(f"[{request_id}] calling ollama started")
        llm_started = time.perf_counter()
        model_output = await generate_with_ollama(prompt)

        logger.info(
            f"[{request_id}] ollama response received llm_sec={time.perf_counter() - llm_started:.3f} "
            f"output_len={len(model_output) if model_output else 0}"
        )

        try:
            parsed = json.loads(model_output)
            logger.info(f"[{request_id}] model output parsed as JSON successfully")
        except JSONDecodeError:
            logger.warning(f"[{request_id}] model output is not valid JSON, fallback raw_output used")
            parsed = {
                "cleaned_transcript": best_text,
                "summary": None,
                "intent": None,
                "topics": [],
                "action_items": [],
                "customer_sentiment": "unknown",
                "manager_quality_score": 0,
                "notes": ["LLM returned non-JSON output"],
                "raw_output": model_output,
            }

        logger.info(
            f"[{request_id}] /analyze-audio completed successfully total_sec={time.perf_counter() - started_at:.3f}"
        )

        return {
            "status": "ok",
            "model": settings.llm_model_name,
            "transcription": transcription,
            "analysis": {
                "cleaned_transcript": parsed.get("cleaned_transcript"),
                "summary": parsed.get("summary"),
                "intent": parsed.get("intent"),
                "topics": parsed.get("topics") or [],
                "action_items": parsed.get("action_items") or [],
                "customer_sentiment": parsed.get("customer_sentiment"),
                "manager_quality_score": parsed.get("manager_quality_score"),
                "notes": parsed.get("notes") or [],
                "raw_output": parsed.get("raw_output"),
            },
        }

    except HTTPException:
        logger.exception(f"[{request_id}] HTTPException inside /analyze-audio")
        raise
    except Exception as exc:
        logger.exception(f"[{request_id}] /analyze-audio failed: {exc}")
        raise HTTPException(status_code=500, detail=f"Audio analysis failed: {str(exc)}")
    finally:
        if preprocessed_path:
            logger.info(f"[{request_id}] deleting preprocessed file path={preprocessed_path}")
            delete_file_safely(preprocessed_path)

        if file_path:
            logger.info(f"[{request_id}] deleting temp file path={file_path}")
            delete_file_safely(file_path)