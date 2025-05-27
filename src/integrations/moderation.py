from src.ai.moderation_llm import moderation_pipeline

def moderate(text: str, model, tokenizer, device, callback_url, logger) -> dict:
    return moderation_pipeline(text, model, tokenizer, device, callback_url, logger) 
