from src.ai.model_manager import model, tokenizer
from src.version import __version__ as MODEL_VERSION
from datetime import datetime, timedelta
from src.common.logger_config import init_process_logging
import random
import re

logger = init_process_logging("ai")

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
        input_ids = tokenizer(full_prompt, return_tensors="pt").input_ids
        output_ids = model.generate(
            input_ids,
            max_new_tokens=20,  # 짧은 출력 강제
            temperature=0.5,  # 더 일관된 출력
            do_sample=True,
            top_k=50,
            top_p=0.9
        )
        # 출력 디코딩 및 후처리
        gen_tokens = output_ids[0, input_ids.shape[1]:]
        meme_full = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()

        # 랜덤 번호 접두사 제거
        pattern = re.escape(str(rand_num)) + r'[:.] ?'
        meme_clean = re.sub(pattern, '', meme_full, count=1).strip()

        # 첫 번째 문장 추출 및 불필요한 요소 제거
        meme = meme_clean.split('.')[0].split('!')[0].split('?')[0].strip()

        # 추가적인 불필요 요소 (해시태그, 번호 등) 제거
        meme = re.sub(r'#.*$', '', meme).strip()  # 해시태그 제거
        meme = re.sub(r'^\d+\s*', '', meme).strip()  # 번호 제거

        logger.info(f"VoteGenerator 완료: {meme}")

        now = datetime.now()
        open_at = now.replace(hour=10, minute=0, second=0, microsecond=0).isoformat() # 전송 시작 날짜 기준 10시 0분 0초
        closed_at = (now + timedelta(days=7)).replace(microsecond=0).isoformat() # 7일 후
        
        return {
            "content": meme.strip(),
            "imageUrl": "",
            "imageName": "",
            "openAt": open_at,
            "closedAt": closed_at,
            "version": MODEL_VERSION
        }
