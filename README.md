# LLM Audio Wrapper API
 
FastAPI wrapper for:
- faster-whisper transcription
- Ollama + Qwen2.5:14b analysis
 
## Main endpoints
- GET /health
- POST /transcribe
- POST /analyze-audio
 
## Run locally
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8001 --reload
 
## Example
curl -X POST http://localhost:8001/analyze-audio \
  -F "file=@sample.wav" 

