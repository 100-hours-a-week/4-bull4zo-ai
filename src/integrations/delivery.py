import requests

class Delivery:
    @staticmethod
    def push(word_id: int, vote: dict):
        # 실제 백엔드 API URL (예시)
        backend_url = f"http://your-backend/api/v1/ai/votes"
        payload = {
            "wordId": word_id,
            "content": vote["content"],
            "imageUrl": None
        }
        print(payload)
        # try:
        #     response = requests.post(backend_url, json=payload)
        #     status_code = response.status_code
        #     try:
        #         body = response.json()
        #         message = body.get("message")
        #         data = body.get("data")
        #     except Exception:
        #         message = None
        #         data = None

        #     # 상태코드별 로그
        #     if status_code == 201:
        #         print("[Delivery] 투표 생성 성공")
        #     elif status_code == 400:
        #         print("[Delivery] 요청 오류(필수값 누락 등)")
        #     elif status_code == 500:
        #         print("[Delivery] 서버 내부 오류")
        #     else:
        #         print(f"[Delivery] 기타 응답: status={status_code}, message={message}")

        #     return status_code, message, data
        # except requests.RequestException as e:
        #     print(f"[Delivery] 백엔드 연결 실패: {e}")
        #     return 500, "UNEXPECTED_ERROR", None 