from multiprocessing import Event, Manager, Process, Queue
import logging
import os
import signal
import sys
from dotenv import load_dotenv
from src.ai.ai_process import run_model_process
from src.api.api_process import run_fastapi_process
from tests.dummy_server import run_dummy_server_process

load_dotenv()
environment = os.getenv("ENVIRONMENT").lower()

if environment not in ["dev", "publish"]:
    raise Exception("environment must be set 'dev' or 'publish'")

def main():
    logging.basicConfig(level=logging.INFO)
    logging.info("Starting AI Server")
    logging.info(f"Environment : {environment}")

    if environment.lower() == "dev":
        dummy_server_process = Process(target=run_dummy_server_process)
        logging.info("Starting Dummy Server Process")
        dummy_server_process.start()

    # 프로세스간 공유할 Queue 생성
    moderation_queue: Queue = Queue()
    result_queue: Queue = Queue()

    # Stop Event 생성
    manager = Manager()
    stop_event = manager.Event()

    # 모델 프로세스 생성 및 시작
    model_process = Process(target=run_model_process, args=(stop_event, moderation_queue, result_queue,))
    logging.info("Starting Model Process")
    model_process.start()

    # API Server 프로세스 생성 및 시작
    fastapi_process = Process(target=run_fastapi_process, args=(moderation_queue, result_queue,))
    logging.info("Starting FastAPI Process")
    fastapi_process.start()

    logging.info("Started AI Server")

    # 프로세스 종료 핸들러
    def shutdown_handler(sig, frame):
        logging.info("Shutdown signal received, terminating processes...")
        logging.info("Stopping AI Server")
        
        if environment.lower() == "dev":
            dummy_server_process.terminate()
            stop_event.set()
            fastapi_process.terminate()
            
            dummy_server_process.join()
            model_process.join()
            fastapi_process.join()
        else:
            stop_event.set()
            fastapi_process.terminate()

            model_process.join()
            fastapi_process.join()

        logging.info("Stop AI Server")
        sys.exit(0)

    # Ctrl+C(SIGINT) 핸들링
    signal.signal(signal.SIGINT, shutdown_handler)
    signal.pause()  # 메인 프로세스는 여기서 대기

if __name__ == "__main__":
    main()
