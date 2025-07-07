def run_analysis_pipeline(start_date: str, end_date: str, moderation_task_queue, result_queue, logger=None):
    if logger:
        logger.info(f"analysis pipeline 시작: start_date[{start_date}] ~ end_date[{end_date}]")
    moderation_task_queue.put({"type": "analysis", "data": {"start_date": start_date, "end_date": end_date}})
