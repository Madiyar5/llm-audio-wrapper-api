import json


def build_analysis_prompt(
    transcript: str,
    alternatives: dict | None = None,
    detected_language: str | None = None,
    mixed_language: bool = False,
    language_hint: str | None = None,
    conversation_type: str = "crm_customer_call",
    metadata: dict | None = None,
) -> str:
    payload = {
        "conversation_type": conversation_type,
        "transcript": transcript,
    }

    if detected_language:
        payload["detected_language"] = detected_language

    if mixed_language:
        payload["mixed_language"] = True

    if language_hint:
        payload["language_hint"] = language_hint

    if alternatives:
        non_empty_alts = {k: v for k, v in alternatives.items() if v}
        if non_empty_alts:
            payload["alternatives"] = non_empty_alts

    if metadata:
        non_empty_meta = {k: v for k, v in metadata.items() if v not in (None, "", [], {})}
        if non_empty_meta:
            payload["metadata"] = non_empty_meta

    return f"""
Ты — аналитик клиентских звонков в CRM.

Тебе передаётся готовый транскрипт звонка, а иногда — альтернативные версии и дополнительная информация.
Нужно:
- восстановить максимально вероятную читаемую версию разговора,
- опираться только на данные из входа,
- ничего не выдумывать,
- если часть фразы неясна, помечать её как [неразборчиво],
- сохранить cleaned_transcript в исходном языке разговора,
- все аналитические поля вернуть на русском языке.

Если информации недостаточно, используй "unknown".

Верни ТОЛЬКО валидный JSON строго заданной структуры.

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
manager_quality_score должен быть целым числом от 0 до 10.

Верни только JSON.

Вход:
{json.dumps(payload, ensure_ascii=False, indent=2)}
""".strip()