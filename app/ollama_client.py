import logging
import time

import httpx

from app.config import settings


logger = logging.getLogger(__name__)


async def _call_ollama(prompt: str, options: dict) -> str:
    started = time.perf_counter()

    payload = {
        "model": settings.llm_model_name,
        "prompt": prompt,
        "stream": False,
        "options": options,
    }

    logger.info(
        "ollama call started model=%s prompt_len=%s timeout=%s options=%s",
        settings.llm_model_name,
        len(prompt),
        settings.ollama_timeout,
        options,
    )

    try:
        timeout = httpx.Timeout(settings.ollama_timeout)

        async with httpx.AsyncClient(timeout=timeout) as client:
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
        logger.exception("Ollama call failed")
        raise


async def generate_with_ollama(prompt: str) -> str:
    return await _call_ollama(
        prompt=prompt,
        options={
            "temperature": 0,
            "top_p": 0.9,
            "num_predict": 700,
        },
    )


async def generate_with_ollama_fast(prompt: str) -> str:
    return await _call_ollama(
        prompt=prompt,
        options={
            "temperature": 0,
            "top_p": 0.9,
            "num_predict": 220,
        },
    )