import httpx
from app.config import settings
 
 
async def generate_with_ollama(prompt: str) -> str:
    url = f"{settings.ollama_base_url}/api/generate"
 
    payload = {
        "model": settings.model_name,
        "prompt": prompt,
        "stream": False,
    }
 
    timeout = httpx.Timeout(settings.ollama_timeout)
 
    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.post(url, json=payload)
        response.raise_for_status()
        data = response.json()
 
    return data.get("response", "").strip()

