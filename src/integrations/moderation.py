from src.ai.moderation_llm import run_moderation

def moderate(text: str, model, tokenizer, device, callback_url, logger) -> dict:
    return run_moderation(text, model, tokenizer, device, callback_url, logger) 