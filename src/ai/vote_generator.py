from src.ai.model_manager import model, tokenizer
from src.version import __version__ as MODEL_VERSION
from datetime import datetime, timedelta
from src.common.logger_config import init_process_logging

logger = init_process_logging("ai")

class VoteGenerator:
    def generate(self, word: str, info: str, model, tokenizer) -> dict:
        logger.info(f"VoteGenerator 시작: word={word}, info={info}")
        prompt = f"아래의 단어와 단어에 대한 정보를 보고 이를 활용한 한 줄로 간단한 밈을 만들어주세요.\n단어: {word}\n정보: {info} "
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids
        output = model.generate(input_ids, max_new_tokens=128)
        text = tokenizer.decode(output[0], skip_special_tokens=True)
        logger.info(f"VoteGenerator 완료: {text}")
        
        now = datetime.now()
        open_at = now.replace(hour=10, minute=0, second=0, microsecond=0).isoformat() # 전송 시작 날짜 기준 10시 0분 0초
        closed_at = (now + timedelta(days=7)).replace(microsecond=0).isoformat() # 7일 후
        return {
            "content": text.strip(),
            "imageUrl": "",
            "imageName": "",
            "openAt": open_at,
            "closedAt": closed_at,
            "version": MODEL_VERSION
        }
