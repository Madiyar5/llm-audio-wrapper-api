import logging
import time

import httpx

from app.config import settings


logger = logging.getLogger(__name__)


async def generate_with_ollama(prompt: str) -> str:
    started = time.perf_counter()

    payload = {
        "model": settings.llm_model_name,
        "prompt": prompt,
        "stream": False,
    }

    logger.info(
        "generate_with_ollama called model=%s prompt_len=%s base_url=%s timeout=%s",
        settings.llm_model_name,
        len(prompt),
        settings.ollama_base_url,
        settings.ollama_timeout,
    )

    try:
        timeout = httpx.Timeout(settings.ollama_timeout)

        async with httpx.AsyncClient(timeout=timeout) as client:
            logger.info("Sending request to Ollama /api/generate")
            response = await client.post(
                f"{settings.ollama_base_url}/api/generate",
                json=payload,
            )

        logger.info(
            "Ollama response received status_code=%s req_sec=%.3f",
            response.status_code,
            time.perf_counter() - started,
        )

        response.raise_for_status()

        data = response.json()
        result = data.get("response", "")

        logger.info("Ollama response text extracted output_len=%s", len(result))
        return result

    except Exception:
        logger.exception("generate_with_ollama failed")
        raise