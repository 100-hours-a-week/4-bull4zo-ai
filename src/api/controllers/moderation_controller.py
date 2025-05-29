import logging
import os
from dotenv import load_dotenv
from fastapi import APIRouter, status
from fastapi.responses import JSONResponse
from src.api.dtos.moderation_result_request import ModerationResultRequest
from src.api.dtos.moderation_request import ModerationRequest
import requests
import uuid
import datetime

def get_router(moderation_task_queue, result_queue, logger=None):
    router = APIRouter(prefix="/api/v1", tags=["Moderation"])

    @router.post("/moderation", status_code=status.HTTP_202_ACCEPTED)
    async def moderate(request: ModerationRequest):
        request_id = str(request.voteId) if request.voteId else str(uuid.uuid4())
        logger.info("검열 요청 수신", 
                   extra={
                       "section": "server", 
                       "request_id": request_id,
                       "content": request.content
                   })
        if not request.content.strip():
            error_message = "content must not be null, empty, or whitespace only."
            logger.error(f"잘못된 요청: {error_message}", 
                        extra={"section": "server", "request_id": request_id})
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={
                    "status": "Bad Request",
                    "message": error_message,
                    "data": None
                }
            )
        try:
            moderation_task_queue.put({"type": "moderation", "data": request})
            logger.info(f"검열 요청 큐에 추가 완료 (ID={request_id})", 
                       extra={"section": "server", "request_id": request_id})
            # 비동기: 즉시 응답만 반환
            return {
                "status": "Accepted",
                "message": "content has been queued",
                "data": None
            }
        except Exception as e:
            error_details = f"검열 요청 처리 중 오류 발생: {str(e)}"
            logger.error(error_details, exc_info=True, 
                        extra={"section": "server", "request_id": request_id})
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={
                    "status": "Internal Server Error",
                    "message": str(e),
                    "data": None
                }
            )

    @router.post("/moderation/test")  
    async def moderation_test(request: ModerationRequest):
        # 요청 ID 생성 (없으면 UUID 사용)
        request_id = str(request.voteId) if request.voteId else str(uuid.uuid4())
        # 요청 로깅
        logger.info("moderation/test 엔드포인트 호출", 
                   extra={
                       "section": "server", 
                       "request_id": request_id,
                       "content": request.content
                   })
        # 입력 검증
        if not request.content.strip():
            error_message = "content must not be null, empty, or whitespace only."
            logger.error(f"잘못된 요청: {error_message}", 
                        extra={"section": "server", "request_id": request_id})
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={
                    "status": "Bad Request",
                    "message": error_message,
                    "data": None
                }
            )
        try:
            logger.info(f"moderation/test 정상 수신 (ID={request_id})", 
                       extra={"section": "server", "request_id": request_id})
            return {
                "status": "Accepted",
                "message": "content has been queued",
                "data": None
            }
        except Exception as e:
            error_details = f"moderation/test 처리 중 오류 발생: {str(e)}"
            logger.error(error_details, exc_info=True, 
                        extra={"section": "server", "request_id": request_id})
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={
                    "status": "Internal Server Error",
                    "message": str(e),
                    "data": None
                }
            )

    @router.post("/vote_generate", status_code=status.HTTP_202_ACCEPTED)
    async def vote_generate(request: dict):
        request_id = str(uuid.uuid4())
        logger.info("투표 생성 요청 수신", 
            extra={
                "section": "server", 
                "request_id": request_id,
                "content": str(request)
            })
        try:
            moderation_task_queue.put({"type": "vote", "data": request})
            logger.info(f"투표 생성 요청 큐에 추가 완료 (ID={request_id})", 
                       extra={"section": "server", "request_id": request_id})
            # 비동기: 즉시 응답만 반환
            return {
                "status": "Accepted",
                "message": "vote generation has been queued",
                "data": None
            }
        except Exception as e:
            error_details = f"투표 생성 요청 처리 중 오류 발생: {str(e)}"
            logger.error(error_details, exc_info=True, 
                        extra={"section": "server", "request_id": request_id})
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={
                    "status": "Internal Server Error",
                    "message": str(e),
                    "data": None
                }
            )

    return router
