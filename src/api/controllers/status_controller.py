from fastapi import APIRouter, status

status_router = APIRouter()

@status_router.get("/status", status_code=status.HTTP_200_OK)
# TODO: Status에 맞게 업데이트 필요
async def get_status():
    #  TODO: 로그 남기기
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
