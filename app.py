from multiprocessing import Event, Manager, Process, Queue
import logging
import signal
import sys
from src.ai.ai_process import run_model_process
from src.api.api_process import run_fastapi_process

def main():
    logging.basicConfig(level=logging.INFO)
    logging.info("Starting AI Server")
    
    # 프로세스간 공유할 Queue 생성
    moderation_queue: Queue = Queue()

    # Stop Event 생성
    manager = Manager()
    stop_event = manager.Event()

    # 프로세스 생성 with Queue (공유 메모리)
    
    model_process = Process(target=run_model_process, args=(stop_event, moderation_queue,))
    fastapi_process = Process(target=run_fastapi_process, args=(moderation_queue,))

    # 프로세스 시작
    logging.info("Starting Model Process")
    model_process.start()
    logging.info("Starting FastAPI Process")
    fastapi_process.start()

    logging.info("Started AI Server")

    # 프로세스 종료 핸들러
    def shutdown_handler(sig, frame):
        logging.info("Shutdown signal received, terminating processes...")
        logging.info("Stopping AI Server")
        
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
