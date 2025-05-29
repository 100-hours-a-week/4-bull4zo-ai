from fastapi import APIRouter, BackgroundTasks, Request
from src.api.dtos.word_create_request import WordCreateRequest
from src.services.word_pipeline import run_pipeline
import json

def get_word_router(moderation_task_queue, result_queue):
    router = APIRouter(prefix="/api/v1", tags=["Words"])
    word_id_counter = {"value": 1}  # mutable dict로 클로저에서 값 변경

    @router.post("/words")
    async def create_word(request: Request, body: WordCreateRequest, background_tasks: BackgroundTasks):
        word_id = word_id_counter["value"]
        word_id_counter["value"] += 1
        # logger 가져오기
        logger = request.app.state.logger
        # moderation.log에 기록
        logger.info(
            json.dumps({
                "word_id": word_id,
                "word": body.word,
                "client_host": request.client.host
            }),
            extra={"section": "moderation", "request_id": str(word_id)}
        )
        background_tasks.add_task(run_pipeline, word_id, body.word, moderation_task_queue, result_queue)
        return {"word_id": word_id, "status": "queued"}

    return router 
