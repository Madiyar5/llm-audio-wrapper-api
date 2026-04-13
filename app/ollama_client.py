import logging
import time
import httpx

from app.config import settings

logger = logging.getLogger(__name__)

FAST_ANALYSIS_SCHEMA = {
    "type": "object",
    "properties": {
        "call_topic": {"type": "string"},
        "call_purpose": {"type": "string"},
        "customer_request": {"type": "string"},
        "product_or_service": {"type": "string"},
        "price_discussed": {"type": "boolean"},
        "call_outcome": {"type": "string"},
        "customer_sentiment": {
            "type": "string",
            "enum": ["positive", "neutral", "negative", "mixed", "unknown"]
        },
        "analysis_confidence": {
            "type": "string",
            "enum": ["high", "medium", "low"]
        },
        "manager_quality_score": {
            "type": "integer",
            "minimum": 0,
            "maximum": 10
        }
    },
    "required": [
        "call_topic",
        "call_purpose",
        "customer_request",
        "product_or_service",
        "price_discussed",
        "call_outcome",
        "customer_sentiment",
        "analysis_confidence",
        "manager_quality_score"
    ]
}


FULL_ANALYSIS_SCHEMA = {
    "type": "object",
    "properties": {
        "call_topic": {"type": "string"},
        "call_purpose": {"type": "string"},
        "customer_request": {"type": "string"},
        "product_or_service": {"type": "string"},
        "key_points": {
            "type": "array",
            "items": {"type": "string"}
        },
        "objections": {
            "type": "array",
            "items": {"type": "string"}
        },
        "price_discussed": {"type": "boolean"},
        "price_details": {"type": "string"},
        "next_step": {"type": "string"},
        "call_outcome": {"type": "string"},
        "customer_sentiment": {
            "type": "string",
            "enum": ["positive", "neutral", "negative", "mixed", "unknown"]
        },
        "manager_quality_score": {
            "type": "integer",
            "minimum": 0,
            "maximum": 10
        },
        "transcript_quality": {
            "type": "string",
            "enum": ["high", "medium", "low"]
        },
        "analysis_confidence": {
            "type": "string",
            "enum": ["high", "medium", "low"]
        },
        "notes": {
            "type": "array",
            "items": {"type": "string"}
        }
    },
    "required": [
        "call_topic",
        "call_purpose",
        "customer_request",
        "product_or_service",
        "key_points",
        "objections",
        "price_discussed",
        "price_details",
        "next_step",
        "call_outcome",
        "customer_sentiment",
        "manager_quality_score",
        "transcript_quality",
        "analysis_confidence",
        "notes"
    ]
}


async def _chat_with_ollama(prompt: str, schema: dict, options: dict) -> str:
    started = time.perf_counter()

    payload = {
        "model": settings.llm_model_name,
        "messages": [
            {
                "role": "user",
                "content": prompt,
            }
        ],
        "stream": False,
        "format": schema,
        "keep_alive": settings.ollama_keep_alive,
        "options": options,
    }

    logger.info(
        "ollama chat started model=%s prompt_len=%s timeout=%s keep_alive=%s options=%s",
        settings.llm_model_name,
        len(prompt),
        settings.ollama_timeout,
        settings.ollama_keep_alive,
        options,
    )

    try:
        timeout = httpx.Timeout(settings.ollama_timeout)

        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(
                f"{settings.ollama_base_url}/api/chat",
                json=payload,
            )

        logger.info(
            "Ollama response received status_code=%s req_sec=%.3f",
            response.status_code,
            time.perf_counter() - started,
        )

        response.raise_for_status()

        data = response.json()
        message = data.get("message", {})
        result = message.get("content", "")

        logger.info("Ollama response text extracted output_len=%s", len(result))
        return result

    except Exception:
        logger.exception("Ollama call failed")
        raise


async def generate_with_ollama(prompt: str) -> str:
    return await _chat_with_ollama(
        prompt=prompt,
        schema=FULL_ANALYSIS_SCHEMA,
        options={
            "temperature": 0,
            "top_p": 0.9,
            "num_predict": 420,
            "num_ctx": 2048,
            "repeat_penalty": 1.05,
        },
    )


async def generate_with_ollama_fast(prompt: str) -> str:
    return await _chat_with_ollama(
        prompt=prompt,
        schema=FAST_ANALYSIS_SCHEMA,
        options={
            "temperature": 0,
            "top_p": 0.9,
            "num_predict": 420,
            "num_ctx": 1024,
            "repeat_penalty": 1.08,
        },
    )