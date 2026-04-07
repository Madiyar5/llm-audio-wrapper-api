import json
from json import JSONDecodeError
 
from fastapi import APIRouter, File, HTTPException, UploadFile
 
from app.config import settings
from app.ollama_client import generate_with_ollama
from app.prompts import build_analysis_prompt
from app.transcription import transcribe_audio
from app.utils import delete_file_safely, save_upload_file
 
router = APIRouter(tags=["analysis"])
 
 
@router.get("/health")
async def health():
    return {
        "status": "ok",
        "app": settings.app_name,
        "model": settings.model_name,
    }
 
 
@router.post("/transcribe")
async def transcribe_endpoint(file: UploadFile = File(...)):
    file_path = None
    try:
        file_path = await save_upload_file(file)
        transcription = transcribe_audio(file_path)
        return {
            "status": "ok",
            "transcription": transcription,
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(exc)}")
    finally:
        if file_path:
            delete_file_safely(file_path)
 
 
@router.post("/analyze-audio")
async def analyze_audio(file: UploadFile = File(...)):
    file_path = None
    try:
        file_path = await save_upload_file(file)
 
        transcription = transcribe_audio(file_path)
        transcript_text = transcription.get("text", "").strip()
        transcript_language = transcription.get("language", "ru")
 
        if not transcript_text:
            raise HTTPException(status_code=400, detail="Empty transcription result")
 
        prompt = build_analysis_prompt(
            transcript=transcript_text,
            language=transcript_language,
            conversation_type="crm_customer_call",
        )
 
        model_output = await generate_with_ollama(prompt)
 
        try:
            parsed = json.loads(model_output)
        except JSONDecodeError:
            parsed = {
                "summary": None,
                "intent": None,
                "topics": [],
                "action_items": [],
                "customer_sentiment": None,
                "manager_quality_score": None,
                "raw_output": model_output,
            }
 
        return {
            "status": "ok",
            "model": settings.model_name,
            "transcription": transcription,
            "analysis": {
                "summary": parsed.get("summary"),
                "intent": parsed.get("intent"),
                "topics": parsed.get("topics") or [],
                "action_items": parsed.get("action_items") or [],
                "customer_sentiment": parsed.get("customer_sentiment"),
                "manager_quality_score": parsed.get("manager_quality_score"),
                "raw_output": parsed.get("raw_output"),
            },
        }
 
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Audio analysis failed: {str(exc)}")
    finally:
        if file_path:
            delete_file_safely(file_path)

