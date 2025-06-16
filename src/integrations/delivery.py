import requests
import os
from typing import Tuple, Optional
from src.api.dtos.model_vote_request import ModelVoteRequest

class Delivery:
    @staticmethod
    def send_model_vote(
        word_id: int,
        vote: dict,
        logger,
        request_id: str,
        backend_url: str
    ) -> Tuple[int, Optional[str], Optional[dict]]:
        payload = ModelVoteRequest(
            content=vote.get("content", ""),
            imageUrl=vote.get("imageUrl", ""),
            imageName=vote.get("imageName", ""),
            openAt=vote.get("openAt", ""),
            closedAt=vote.get("closedAt", ""),
            version=vote.get("version", "")
        )
        headers = {"Content-Type": "application/json"}
        logger.info(f"모델 투표 결과 전송 시작", extra={"section": "server", "request_id": request_id, "word_id": word_id})
        try:
            # Pydantic v2 이상 호환: model_dump() 사용
            payload_dict = payload.model_dump() if hasattr(payload, 'model_dump') else payload.dict()
            response = requests.post(backend_url, json=payload_dict, headers=headers)
            status_code = response.status_code
            try:
                body = response.json()
                message = body.get("message")
                data = body.get("data")
            except Exception:
                message = None
                data = None

            if status_code == 201:
                logger.info(
                    f"모델 투표 결과 전송 성공: HTTP {status_code}, wordId={word_id}, message={message}",
                    extra={"section": "server", "request_id": request_id}
                )
            else:
                logger.error(
                    f"모델 투표 결과 전송 실패: HTTP {status_code}, wordId={word_id}, message={message}",
                    extra={"section": "server", "request_id": request_id}
                )
            return status_code, message, data
        except requests.exceptions.RequestException as e:
            error_msg = f"모델 투표 결과 전송 중 네트워크 오류: {str(e)}"
            logger.error(error_msg, exc_info=True, extra={"section": "server", "request_id": request_id})
            return 500, "UNEXPECTED_ERROR", None

    @staticmethod
    def send_moderation_callback(moderation_result_request, callback_url, logger, request_id):
        headers = {"Content-Type": "application/json"}
        try:
            logger.info(f"일반 검열 결과 전송 시작", extra={"section": "server", "request_id": request_id})
            response = requests.post(callback_url, json=moderation_result_request.dict(), headers=headers)
            if response.status_code == 201:
                logger.info(f"일반 검열 결과 전송 성공: HTTP {response.status_code}", extra={"section": "server", "request_id": request_id})
            else:
                logger.error(f"일반 검열 결과 전송 실패: HTTP {response.status_code}", extra={"section": "server", "request_id": request_id})
        except requests.exceptions.RequestException as e:
            error_msg = f"일반 검열 결과 전송 중 네트워크 오류: {str(e)}"
            logger.error(error_msg, exc_info=True, extra={"section": "server", "request_id": request_id}) 
