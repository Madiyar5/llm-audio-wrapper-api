def build_analysis_prompt(transcript: str, language: str = "ru", conversation_type: str = "crm_customer_call") -> str:
    return f"""
You are an assistant for CRM customer conversation analysis.
 
Task:
Analyze the following transcript and return ONLY valid JSON.
Do not include markdown.
Do not include explanations.
Do not wrap the response in triple backticks.
 
Required JSON schema:
{{
  "summary": "short summary of the conversation",
  "intent": "main customer intent",
  "topics": ["topic1", "topic2"],
  "action_items": ["action1", "action2"],
  "customer_sentiment": "positive | neutral | negative",
  "manager_quality_score": 1
}}
 
Rules:
- manager_quality_score must be an integer from 1 to 10
- topics must be an array of short strings
- action_items must be an array of short strings
- summary must be concise and useful
- intent must be a short machine-readable label
- language of the values should match the transcript language when possible
 
Conversation type: {conversation_type}
Language: {language}
 
Transcript:
{transcript}
""".strip()

