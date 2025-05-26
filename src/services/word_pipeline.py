from src.integrations.info_fetcher import InfoFetcher
from src.integrations.moderation import moderate
from src.ai.vote_generator import VoteGenerator
from src.integrations.delivery import Delivery
from src.ai.model_manager import model, tokenizer, load_model
import os

def run_pipeline(word_id: int, word: str, moderation_queue, result_queue):
    # 모델 직접 사용 X, 큐에 요청만 넣음
    moderation_queue.put({"type": "vote", "data": {"word_id": word_id, "word": word}})
    # 필요시 result_queue에서 결과를 받을 수 있음 (비동기라면 생략)

def run_pipeline_old(word_id: int, word: str):
    global model, tokenizer
    if model is None or tokenizer is None:
        model_name = "naver-hyperclovax/HyperCLOVAX-SEED-Vision-Instruct-3B"
        hf_token = os.getenv("HF_TOKEN")
        model, tokenizer = load_model(model_name, hf_token)
    # 1. 단어/Word 조회 (이미 word 인자로 받음)
    # 2. 정보 fetch
    info = InfoFetcher.fetch(word)
    # 3. WordInfo 저장 (생략)
    # 4. 투표 생성
    vote = VoteGenerator().generate(word, info, model, tokenizer)
    # 5. 검열
    mod = moderate(vote["content"])
    if mod["result"] == "REJECTED":
        # ModerationLog 저장 (생략)
        return
    # 6. Vote 저장 (생략)
    # 7. Delivery
    Delivery.push(word_id, vote) 