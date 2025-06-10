def run_pipeline(word_id: int, word: str, moderation_task_queue, result_queue, logger=None):
    if logger:
        logger.info(f"word pipeline 시작: word_id={word_id}, word={word}")
    # 모델 직접 사용 X, 큐에 요청만 넣음
    moderation_task_queue.put({"type": "vote", "data": {"word_id": word_id, "word": word}})
    # 필요시 result_queue에서 결과를 받을 수 있음 (비동기라면 생략)
