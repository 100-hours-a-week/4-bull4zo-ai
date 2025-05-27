import requests

class Delivery:
    @staticmethod
    def push(word_id: int, vote: dict, logger, request_id: str):
        backend_url = f"http://your-backend/api/v1/ai/votes"
        payload = {
            "wordId": word_id,
            "content": vote["content"],
            "imageUrl": None
        }
        headers = {"Content-Type": "application/json"}
        try:
            logger.info(f"투표 결과 전송 시작", extra={"section": "server", "request_id": request_id})
            response = requests.post(backend_url, json=payload, headers=headers)
            status_code = response.status_code
            try:
                body = response.json()
                message = body.get("message")
                data = body.get("data")
            except Exception:
                message = None
                data = None
            if status_code == 201:
                logger.info(f"투표 결과 전송 성공: HTTP {status_code}", extra={"section": "server", "request_id": request_id})
            elif status_code == 400:
                logger.error(f"투표 결과 전송 실패: HTTP {status_code}", extra={"section": "server", "request_id": request_id})
            elif status_code == 404:
                logger.error(f"투표 결과 전송 실패: HTTP {status_code}", extra={"section": "server", "request_id": request_id})
            elif status_code == 500:
                logger.error(f"투표 결과 전송 실패: HTTP {status_code}", extra={"section": "server", "request_id": request_id})
            else:
                logger.error(f"투표 결과 전송 실패: HTTP {status_code}", extra={"section": "server", "request_id": request_id})
            return status_code, message, data
        except requests.exceptions.RequestException as e:
            error_msg = f"투표 결과 전송 중 네트워크 오류: {str(e)}"
            logger.error(error_msg, exc_info=True, extra={"section": "server", "request_id": request_id})
            return 500, "UNEXPECTED_ERROR", None 
