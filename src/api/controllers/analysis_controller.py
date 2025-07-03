import json
from fastapi import APIRouter, BackgroundTasks, status, Request
from api.dtos.analysis_request import AnalysisRequest
from services.analysis_pipeline import run_analysis_pipeline

def get_word_router(moderation_task_queue, result_queue):
    router = APIRouter(prefix="/api/v1", tags=["Analysis"])

    @router.post("/analysis")
    async def create_word(request: Request, body: AnalysisRequest, background_tasks: BackgroundTasks):

        # logger 가져오기
        logger = request.app.state.logger
        # moderation.log에 기록
        logger.info(
            json.dumps({
                "method": "analysis",
                "start_date": body.start_date,
                "end_date": body.end_date,
                "client_host": request.client.host
            }),
            extra={"section": "analysis"}
        )

        # Queue에 데이터 삽입
        background_tasks.add_task(run_analysis_pipeline, body.start_date, body.end_date, moderation_task_queue, result_queue, logger)
        return {"method": "alanysis", "status": "queued"}

    return router 
