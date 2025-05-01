import os
from dotenv import load_dotenv
from fastapi import FastAPI, status
import uvicorn
from src.api.dtos.moderation_result_request import ModerationResultRequest

load_dotenv()
be_server_port = os.getenv("BE_SERVER_PORT")

def run_dummy_server_process():
    app = FastAPI(
        title="Dummy Server",
        description="MOA Project - Dummy Server (for BE Server)",
        version="1.0.0"
    )
    
    @app.post("/api/v1/ai/votes/moderation/callback", status_code=status.HTTP_201_CREATED)
    def create_moderated_vote(moderation_result: ModerationResultRequest):
        print(moderation_result)
        return {
            "message": "SUCCESS",
            "data": {
                "voteId": moderation_result.voteId,
                "stored": True
            }
        }
    
    uvicorn.run(app, host="0.0.0.0", port=int(be_server_port))
