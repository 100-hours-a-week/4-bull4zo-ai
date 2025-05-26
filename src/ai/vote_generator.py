from src.ai.model_manager import model, tokenizer

class VoteGenerator:
    def generate(self, word: str, info: str, model, tokenizer) -> dict:
        prompt = f"단어: {word}\n정보: {info}\n 를 보고 간단한 찬반 투표를 생성해주세요."
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids
        output = model.generate(input_ids, max_new_tokens=128)
        text = tokenizer.decode(output[0], skip_special_tokens=True)
        return {"content": text.strip()}