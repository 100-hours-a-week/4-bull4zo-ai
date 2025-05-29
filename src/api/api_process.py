from multiprocessing import Queue
from multiprocessing.synchronize import Event
from fastapi import FastAPI, Request
import uvicorn
from src.api.controllers.moderation_controller import get_router
from src.api.controllers.status_controller import status_router
from src.api.controllers.word_controller import get_word_router
from src.common.logger_config import init_process_logging, shutdown_logging
import datetime
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_fastapi_instrumentator.metrics import latency, requests
from prometheus_client import Gauge
from starlette.middleware.base import BaseHTTPMiddleware

INPROGRESS_REQUESTS = Gauge(
    "inprogress_requests",
    "Number of in-progress requests"
)

def run_fastapi_process(moderation_task_queue: Queue, result_queue: Queue):
    # API 로거 초기화
    logger = init_process_logging("api")
    logger.info("API 서버 프로세스 시작", extra={"section": "server", "request_id": "init"})
    print(f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S,%f')[:-3]} [INFO] server:init - API 서버 프로세스 시작")
    
    app = FastAPI(
        title="AI Server API",
        description="MOA Project - AI Server (HyperCLOVAX 기반 AI Service)",
        version="1.0.0"
    )

    class InProgressMiddleware(BaseHTTPMiddleware):
        async def dispatch(self, request: Request, call_next):
            INPROGRESS_REQUESTS.inc()
            try:
                response = await call_next(request)
            finally:
                INPROGRESS_REQUESTS.dec()
            return response

    app.add_middleware(InProgressMiddleware)

    # 로거를 FastAPI 앱 상태에 저장하여 컨트롤러에서 접근 가능하게 함
    app.state.logger = logger
    
    # 라우터 설정
    app.include_router(get_router(moderation_task_queue, result_queue, logger))
    app.include_router(status_router)
    app.include_router(get_word_router(moderation_task_queue, result_queue))
    
    # Prometheus Instrumentator 설정 (latency, requests만 추가)
    instrumentator = (
        Instrumentator()
        .add(latency())
        .add(requests())
    )
    instrumentator.instrument(app).expose(app, include_in_schema=False)
    
    logger.info("API 서버 시작 준비 완료", extra={"section": "server", "request_id": "init"})
    print(f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S,%f')[:-3]} [INFO] server:init - API 서버 시작 준비 완료")
    
    # uvicorn 서버 실행
    try:
        uvicorn.run(app, host="0.0.0.0", port=8000)
    except Exception as e:
        logger.error(f"서버 실행 중 오류 발생: {str(e)}", exc_info=True, 
                    extra={"section": "server", "request_id": "error"})
        print(f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S,%f')[:-3]} [ERROR] server:error - 서버 실행 중 오류 발생: {str(e)}")
    finally:
        # 종료 시 로깅 시스템 정리
        logger.info("API 서버 종료", extra={"section": "server", "request_id": "shutdown"})
        print(f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S,%f')[:-3]} [INFO] server:shutdown - API 서버 종료")
        shutdown_logging("api")
