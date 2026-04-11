import json


def build_analysis_prompt(
    transcript: str,
    alternatives: dict | None,
    detected_language: str | None,
    mixed_language: bool = False,
    language_hint: str | None = None,
    conversation_type: str = "crm_customer_call",
    metadata: dict | None = None,
) -> str:
    payload = {
        "detected_language": detected_language,
        "mixed_language": mixed_language,
        "language_hint": language_hint,
        "transcript": transcript,
        "alternatives": alternatives or {},
        "conversation_type": conversation_type,
        "metadata": metadata or {},
    }

    return f"""
Ты — аналитик клиентских звонков в CRM.

Тебе передаются:
1) основной транскрипт,
2) возможные альтернативные версии транскрипта,
3) информация о языке,
4) дополнительные метаданные.

Твоя задача:
- восстановить максимально вероятную читаемую версию разговора,
- использовать только ту информацию, которая подтверждается транскриптом,
- ничего не выдумывать,
- если фрагмент неразборчив, помечать его как [неразборчиво],
- сохранить cleaned_transcript в исходном языке разговора,
- все аналитические поля вернуть на русском языке.

Если информации недостаточно, указывай "unknown".

Верни ТОЛЬКО валидный JSON строго такой структуры:

{{
  "cleaned_transcript": "string",
  "call_topic": "string",
  "call_purpose": "string",
  "customer_request": "string",
  "product_or_service": "string",
  "key_points": ["string"],
  "objections": ["string"],
  "price_discussed": true,
  "price_details": "string",
  "next_step": "string",
  "call_outcome": "string",
  "customer_sentiment": "positive|neutral|negative|mixed|unknown",
  "manager_quality_score": 0,
  "transcript_quality": "high|medium|low",
  "analysis_confidence": "high|medium|low",
  "notes": ["string"]
}}

Правила:
- Не выдумывай название продукта, цену или исход звонка, если этого нет в транскрипте.
- Отдавай предпочтение осторожному анализу, а не галлюцинациям.
- manager_quality_score должен быть целым числом от 0 до 10.
- price_discussed должен быть true или false.

Вход:
{json.dumps(payload, ensure_ascii=False, indent=2)}
""".strip()


def build_fast_analysis_prompt(
    transcript: str,
    conversation_type: str = "crm_customer_call",
) -> str:
    payload = {
        "conversation_type": conversation_type,
        "transcript": transcript,
    }

    return f"""
Ты — аналитик клиентских звонков.

Ниже дан уже готовый транскрипт звонка.
Нужно вернуть только краткий аналитический JSON на русском языке.
Не выдумывай факты.
Если информации недостаточно, пиши "unknown".

Верни только JSON.

Вход:
{json.dumps(payload, ensure_ascii=False, indent=2)}
""".strip()