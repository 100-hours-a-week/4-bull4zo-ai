import json
from fastapi import APIRouter, BackgroundTasks, status, Request
import pytz
from api.dtos.analysis_request import AnalysisRequest
from services.analysis_pipeline import run_analysis_pipeline
from datetime import datetime, timedelta

def get_word_router(moderation_task_queue, result_queue):
    router = APIRouter(prefix="/api/v1", tags=["Analysis"])

    @router.post("/analysis")
    async def analyze(request: Request, background_tasks: BackgroundTasks):
        kst = pytz.timezone("Asia/Seoul")
        now_kst = datetime.now(kst)
        today_weekday = now_kst.weekday()
        this_monday_kst = (now_kst - timedelta(days=today_weekday)).replace(hour=0, minute=0, second=0, microsecond=0)
        last_monday_kst = this_monday_kst - timedelta(days=7)
        last_sunday_kst = this_monday_kst - timedelta(days=1)

        last_monday_utc = last_monday_kst.astimezone(pytz.utc)
        last_sunday_utc = last_sunday_kst.astimezone(pytz.utc)

        # ISO8601 포맷 문자열 반환 (밀리초/타임존 제외)
        start_date = last_monday_utc.strftime("%Y-%m-%dT%H:%M:%S")
        end_date = last_sunday_utc.strftime("%Y-%m-%dT%H:%M:%S")

        # logger 가져오기
        logger = request.app.state.logger
        logger.info(
            json.dumps({
                "method": "analysis",
                "start_date": start_date,
                "end_date": end_date,
                "client_host": request.client.host
            }),
            extra={"section": "analysis"}
        )

        # Queue에 데이터 삽입
        background_tasks.add_task(run_analysis_pipeline, start_date, end_date, moderation_task_queue, result_queue, logger)
        return {"method": "alanysis", "status": "queued"}

    return router 
