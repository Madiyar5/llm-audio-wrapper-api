import json


def build_analysis_prompt(
    best_text: str,
    alternatives: dict,
    detected_language: str | None,
    conversation_type: str = "crm_customer_call",
) -> str:
    payload = {
        "detected_language": detected_language,
        "best_text": best_text,
        "alternatives": alternatives,
        "conversation_type": conversation_type,
    }

    return f"""
You are an expert speech-analytics assistant for CRM customer calls.

You will receive:
1) the best automatic transcript hypothesis,
2) alternative transcript hypotheses for the same audio,
3) the detected language.

Your job:
- reconstruct the most probable readable transcript,
- keep the original meaning,
- do NOT invent facts that are not supported by at least one transcript version,
- if a fragment is unclear, mark it as [неразборчиво],
- preserve Russian and Kazakh text as-is,
- then analyze the conversation.

Return ONLY valid JSON with this exact structure:

{{
  "cleaned_transcript": "string",
  "summary": "string",
  "intent": "string",
  "topics": ["string"],
  "action_items": ["string"],
  "customer_sentiment": "positive|neutral|negative|mixed|unknown",
  "manager_quality_score": 0,
  "notes": ["string"]
}}

Input:
{json.dumps(payload, ensure_ascii=False, indent=2)}
""".strip()