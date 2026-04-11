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
You are an expert call-analysis assistant for CRM customer phone calls.

You will receive:
1) the main transcript,
2) optional alternative transcript hypotheses,
3) detected language info,
4) optional metadata.

Your job:
- reconstruct the most probable readable transcript,
- keep only information supported by the transcript hypotheses,
- do NOT invent facts,
- if a fragment is unclear, mark it as [неразборчиво],
- preserve Russian and Kazakh text as-is in cleaned_transcript,
- then analyze the phone call.

Language rules:
- cleaned_transcript must stay in the original language of the conversation,
- ALL analytical fields must be returned in Russian,
- if there is not enough evidence, use "unknown".

Return ONLY valid JSON with this exact structure:

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

Rules:
- If transcript quality is poor, explicitly say so.
- Do not guess product names, prices, or outcomes unless supported by transcript text.
- Prefer conservative analysis over hallucination.
- manager_quality_score must be an integer from 0 to 10.
- price_discussed must be true or false.

Input:
{json.dumps(payload, ensure_ascii=False, indent=2)}
""".strip()