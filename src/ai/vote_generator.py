from src.ai.model_manager import model, tokenizer
from src.version import __version__ as MODEL_VERSION
from datetime import datetime, timedelta, timezone
from src.common.logger_config import init_process_logging
import random
import re

logger = init_process_logging("ai")

KST = timezone(timedelta(hours=9))

class VoteGenerator:
    def generate(self, word: str, info: str, model, tokenizer) -> dict:
        logger.info(f"VoteGenerator 시작: word={word}, info={info}")

        system_prompt = (
            "당신은 '밈 생성 봇'입니다. 찬반 의견이 갈리는 주제에 대해 "
            "짧고 간결한 한 줄 밈을 생성하세요. 설명, 해시태그, 번호, 따옴표 등은 포함하지 말고, "
            "오직 밈 문장 한 줄만 출력하세요."
        )

        rand_num = random.randint(1, 10000)
        
        user_prompt = f"단어: {word}\n정보: {info}\n랜덤 번호: {rand_num}\n밈:"
        full_prompt = f"<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n{user_prompt} [/INST]"

        # 토크나이즈 및 생성
        inputs = tokenizer(full_prompt, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        input_ids = inputs["input_ids"]
        input_length = input_ids.shape[1]

        output_ids = model.generate(
            **inputs, # Pass the dictionary output from tokenizer
            max_new_tokens=512,  # 짧은 출력 강제
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id # Add pad_token_id
        )        
        generated_ids = output_ids[0][input_length:]                     # 프롬프트 이후 토큰만 슬라이스
        meme = tokenizer.decode(generated_ids, skip_special_tokens=True) # 특수 토큰 없이 디코딩
        meme = meme.strip()                                             # 앞뒤 공백/줄바꿈 제거

        # 혹시 여전히 여러 줄이 나올 때, 마지막 줄만 취하고 싶으면:
        meme_line = meme.splitlines()[-1]
        meme_end = re.split(r"[\n#]", meme_line)[0].strip()

        print(meme_end)

        logger.info(f"VoteGenerator 완료: {meme_end}")

        now = datetime.now(KST)
        open_at = now.replace(hour=10, minute=0, second=0, microsecond=0, tzinfo=None).isoformat(timespec='seconds') # 전송 시작 날짜 기준 10시 0분 0초
        closed_at = (now + timedelta(days=7)).replace(microsecond=0, tzinfo=None).isoformat(timespec='seconds') # 7일 후
        
        return {
            "content": meme_end.strip(),
            "imageUrl": "",
            "imageName": "",
            "openAt": open_at,
            "closedAt": closed_at
        }
