from faster_whisper import WhisperModel
from app.config import settings
 
_whisper_model = None
 
 
def get_whisper_model():
    global _whisper_model
    if _whisper_model is None:
        _whisper_model = WhisperModel(
            settings.whisper_model_size,
            device="cpu",
            compute_type=settings.whisper_compute_type,
        )
    return _whisper_model
 
 
def transcribe_audio(file_path: str):
    model = get_whisper_model()
    segments, info = model.transcribe(file_path)
 
    text_parts = []
    for segment in segments:
        text_parts.append(segment.text.strip())
 
    transcript = " ".join(part for part in text_parts if part).strip()
 
    return {
        "language": info.language,
        "duration": getattr(info, "duration", None),
        "text": transcript,
    }

