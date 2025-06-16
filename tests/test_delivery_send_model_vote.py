from src.integrations.delivery import Delivery
import os
import sys
class SimpleLogger:
    def info(self, *args, **kwargs):
        print("[INFO]", *args)
    def error(self, *args, **kwargs):
        print("[ERROR]", *args)

if __name__ == "__main__":
    word_id = 123
    vote = {
        "content": "AI 밈 생성 테스트",
        "imageUrl": "",
        "imageName": "",
        "openAt": "2024-06-17T10:00:00",
        "closedAt": "2024-06-24T10:00:00"
    }
    logger = SimpleLogger()
    request_id = "test-req-1"
    be_server_ip = os.getenv("BE_SERVER_IP","34.22.77.133")
    be_server_port = os.getenv("BE_SERVER_PORT","8080")
    backend_url = f"http://{be_server_ip}:{be_server_port}/api/v1/ai/votes"

    result = Delivery.send_model_vote(word_id, vote, logger, request_id, backend_url)
    print(result)
