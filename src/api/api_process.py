from multiprocessing import Queue
from multiprocessing.synchronize import Event
from fastapi import FastAPI
import uvicorn
from src.api.controllers.moderation_controller import get_router
from src.api.controllers.status_controller import status_router

def run_fastapi_process(moderation_queue: Queue):
    app = FastAPI(
        title="AI Server API",
        description="MOA Project - AI Server (HyperCLOVAX 기반 AI Service)",
        version="1.0.0"
    )
    app.include_router(get_router(moderation_queue))
    app.include_router(status_router)
    uvicorn.run(app, host="0.0.0.0", port=8000)
