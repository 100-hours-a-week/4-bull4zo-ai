import os
from dotenv import load_dotenv
from src.celery_app import celery_app
from src.ai.vote_generator import VoteGenerator
from src.integrations.info_fetcher import InfoFetcher
from src.api.dtos.moderation_request import ModerationRequest
from src.integrations.moderation import moderate
from src.integrations.delivery import Delivery
from src.common.logger_config import init_process_logging
from src.ai.model_manager import load_model
import threading

load_dotenv()
hf_token = os.getenv("HF_TOKEN")
model_name = os.getenv("MODEL_NAME", "naver-hyperclovax/HyperCLOVAX-SEED-Vision-Instruct-3B")
be_server_ip = os.getenv("BE_SERVER_IP", "localhost")
be_server_port = os.getenv("BE_SERVER_PORT", "8000")
callback_url = f"http://{be_server_ip}:{be_server_port}/api/v1/ai/votes/moderation/callback"
logger = init_process_logging("ai")

vote_lock = threading.Lock()

model, tokenizer = load_model(model_name, hf_token)
device = model.device if hasattr(model, 'device') else ("cuda" if os.environ.get('CUDA_VISIBLE_DEVICES') else "cpu")

@celery_app.task
def generate_and_moderate_vote(word_id, word):
    with vote_lock:
        info = InfoFetcher.fetch(word)
        vote = VoteGenerator().generate(word, info, model, tokenizer)
        moderation_request = ModerationRequest(voteId=word_id, content=vote["content"])
        mod = moderate(moderation_request, model, tokenizer, device, callback_url, logger)
        if mod["result"] == "REJECTED":
            logger.info(f"투표 거절: {mod}", extra={"section": "moderation", "request_id": str(word_id)})
            return {"word_id": word_id, "status": "rejected", "content": vote["content"]}
        else:
            backend_url = f"http://{be_server_ip}:{be_server_port}/api/v1/ai/votes"
            Delivery.send_model_vote(word_id, vote, logger, str(word_id), backend_url=backend_url)
            logger.info(f"투표 전송 완료: {vote['content']}", extra={"section": "server", "request_id": str(word_id)})
            return {"word_id": word_id, "status": "delivered", "content": vote["content"]}

@celery_app.task
def moderate_vote_content(vote_id, content):
    moderation_request = ModerationRequest(voteId=vote_id, content=content)
    mod = moderate(moderation_request, model, tokenizer, device, callback_url, logger)
    if mod["result"] == "REJECTED":
        logger.info(f"투표 거절: {mod}", extra={"section": "moderation", "request_id": str(vote_id)})
    else:
        logger.info(f"투표 승인: {mod}", extra={"section": "moderation", "request_id": str(vote_id)})
    return mod
