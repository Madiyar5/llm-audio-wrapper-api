from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
 
 
class AudioAnalysisResponse(BaseModel):
    status: str
    model: str
    transcription: Dict[str, Any]
    analysis: Dict[str, Any]
 
 
class TranscriptionResponse(BaseModel):
    status: str
    transcription: Dict[str, Any]
 
 
class HealthResponse(BaseModel):
    status: str
    app: str
    model: str

