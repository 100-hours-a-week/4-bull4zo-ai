from fastapi import APIRouter, status, Request

status_router = APIRouter(prefix="/api/v1", tags=["Status"])

@status_router.get("/status", status_code=status.HTTP_200_OK)
# TODO: Status에 맞게 업데이트 필요
async def get_status(request: Request):
    # request 객체에서 로거 가져오기
    logger = request.app.state.logger
    logger.info("서버 상태 확인 요청", extra={"section": "server", "request_id": "status"})
    
    return {
        "status": "OK",
        "message": "",
        "data": {
            "aiServerStatus": [
                {
                    "name": "AI Server #1",
                    "status": "Running",
                    "details": [
                        {
                            "moderationStatus": "Running",
                            "voteCreationStatus": "Idle"
                        }
                    ]
                }
            ]
        }
    }
