import logging
import os
from dotenv import load_dotenv
from fastapi import APIRouter, status
from fastapi.responses import JSONResponse
from src.api.dtos.moderation_result_request import ModerationResultRequest
from src.api.dtos.moderation_request import ModerationRequest
import requests

def get_router(moderation_queue):
    router = APIRouter(prefix="/api/v1", tags=["Moderation"])

    @router.post("/moderation", status_code=status.HTTP_202_ACCEPTED)
    async def moderate(request: ModerationRequest):
        #  TODO: 로그 남기기
        if not request.voteContent.strip():
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={
                    "status": "Bad Request",
                    "message": "voteContent must not be null, empty, or whitespace only.",
                    "data": None
                }
            )

        try:
            moderation_queue.put(request)
            #  TODO: 로그 남기기
            return {
                "status": "Accepted",
                "message": "voteContent has been queued",
                "data": None
            }
        except Exception as e:
            #  TODO: 로그 남기기
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
        load_dotenv()
        be_server_ip = os.getenv("BE_SERVER_IP")
        be_server_port = os.getenv("BE_SERVER_PORT")
        callback_url = f"http://{be_server_ip}:{be_server_port}/api/v1/ai/votes/moderation/callback"

        headers = {
            "Content-Type": "application/json"
        }
        
        moderation_result_request = ModerationResultRequest(
            voteId=123,
            result="REJCTED",
            reason="SPAM",
            reasonDetail="부적절한 표현이 발견되었습니다.",
            version="1.0.0"
        )

        try:
            response = requests.post(callback_url, json = dict(moderation_result_request), headers = headers)
            if response.status_code == 201:
                result = response.json()
                print("[201 Created] 저장 성공:", result)
            elif response.status_code == 400:
                print("[400 Bad Request] 요청이 잘못되었습니다:", response.text)
            elif response.status_code == 404:
                print("[404 Not Found] 경로를 찾을 수 없습니다:", response.text)
            elif response.status_code == 500:
                print("[500 Internal Server Error] 서버 오류:", response.text)
            else:
                print(f"[{response.status_code}] 예상치 못한 응답:", response.text)
        except requests.exceptions.RequestException as e:
            print("⚠️ 요청 중 예외 발생:", e)

    return router
