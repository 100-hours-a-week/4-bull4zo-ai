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

def get_router(moderation_queue, logger=None):
    router = APIRouter(prefix="/api/v1", tags=["Moderation"])

    @router.post("/moderation", status_code=status.HTTP_202_ACCEPTED)
    async def moderate(request: ModerationRequest):
        # 요청 ID 생성 (없으면 UUID 사용)
        request_id = str(request.voteId) if request.voteId else str(uuid.uuid4())
        
        # 요청 로깅
        logger.info("검열 요청 수신", 
                   extra={
                       "section": "server", 
                       "request_id": request_id,
                       "content": request.voteContent
                   })
        
        if not request.voteContent.strip():
            error_message = "voteContent must not be null, empty, or whitespace only."
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
            moderation_queue.put(request)
            logger.info(f"검열 요청 큐에 추가 완료 (ID={request_id})", 
                       extra={"section": "server", "request_id": request_id})
            return {
                "status": "Accepted",
                "message": "voteContent has been queued",
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
    def send_result_test():
        # 고정 ID 사용
        request_id = "1"
        logger.info("검열 결과 테스트 요청 시작", 
                   extra={"section": "server", "request_id": request_id})
        
        load_dotenv()
        be_server_ip = os.getenv("BE_SERVER_IP")
        be_server_port = os.getenv("BE_SERVER_PORT")
        callback_url = f"http://{be_server_ip}:{be_server_port}/api/v1/ai/votes/moderation/callback"

        headers = {
            "Content-Type": "application/json"
        }
        
        moderation_result_request = ModerationResultRequest(
            voteId=1,
            result="REJECTED",
            reason="SPAM",
            reasonDetail="부적절한 표현이 발견되었습니다.",
            version="1.0.0"
        )

        try:
            logger.info(f"검열 결과 전송 시작", 
                       extra={"section": "server", "request_id": request_id})
            
            response = requests.post(callback_url, json = dict(moderation_result_request), headers = headers)
            
            if response.status_code == 201:
                result = response.json()
                # 검열 결과 전송 성공 로그를 CSV 파일에 남김
                logger.info(f"검열 결과 전송 성공: HTTP {response.status_code}", 
                           extra={
                               "section": "server", 
                               "request_id": request_id,
                               "pred_label": "SPAM",  # 테스트용 값 추가
                               "pred_score": "1.0",   # 테스트용 값 추가
                               "model_version": "1.0.0"
                           })
                
                print("[201 Created] 저장 성공:", result)
            elif response.status_code == 400:
                error_msg = f"검열 결과 전송 실패: HTTP {response.status_code}"
                logger.error(error_msg, 
                            extra={"section": "server", "request_id": request_id})
                print(error_msg)
            elif response.status_code == 404:
                error_msg = f"검열 결과 전송 실패: HTTP {response.status_code}"
                logger.error(error_msg, 
                            extra={"section": "server", "request_id": request_id})
                print(error_msg)
            elif response.status_code == 500:
                error_msg = f"검열 결과 전송 실패: HTTP {response.status_code}"
                logger.error(error_msg, 
                            extra={"section": "server", "request_id": request_id})
                print(error_msg)
            else:
                error_msg = f"검열 결과 전송 실패: HTTP {response.status_code}"
                logger.error(error_msg, 
                            extra={"section": "server", "request_id": request_id})
                print(error_msg)
        except requests.exceptions.RequestException as e:
            error_msg = f"요청 중 오류 발생"
            logger.error(error_msg, exc_info=True, 
                        extra={"section": "server", "request_id": request_id})
            print(f"⚠️ {error_msg}: {str(e)}")

    return router
