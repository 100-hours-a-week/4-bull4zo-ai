from fastapi import APIRouter, status
from fastapi.responses import JSONResponse
from src.api.dtos.moderation_request import ModerationRequest

def get_router(moderation_queue):
    router = APIRouter()

    @router.post("/moderate", status_code=status.HTTP_202_ACCEPTED)
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
            moderation_queue.put(request.voteContent)
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

    return router
