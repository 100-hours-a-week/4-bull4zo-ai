import os
import time
import logging
from multiprocessing import Queue
from multiprocessing.synchronize import Event
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer
import torch

load_dotenv()
hf_token = os.getenv("HF_TOKEN")

if not hf_token:
    raise ValueError("HF_TOKEN is not set. Please check your .env file.")

def run_model_process(stop_event: Event, moderation_queue: Queue):
    # Device 설정
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"추론 디바이스: {device}")
    if device == "cuda":
        logging.info(f"   GPU: {torch.cuda.get_device_name(0)}\n")

    # LLM 로딩
    logging.info("Loading model...")

    model_name = "naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-0.5B"

    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        trust_remote_code=True,
        token=hf_token
    ).to(device=device)
    
    processor = AutoProcessor.from_pretrained(
        model_name, 
        trust_remote_code=True,
        token=hf_token
    )
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        token=hf_token
    )
    
    logging.info("Model loaded successfully. Waiting for moderation tasks...")

    logging.info("System Started")

    while not stop_event.is_set():
        if not moderation_queue.empty():
            text = moderation_queue.get()

            try:
                logging.info(f"Processing moderation for: {text}")
                
                chat = [
                    {"role": "system", "content": "당신은 검열 시스템입니다. 입력된 텍스트가 부적절한지 판단해야 합니다. '검열 필요: [이유]' 또는 '검열 불필요: 적절한 표현입니다' 형식으로만 답변하세요."},
                    {"role": "user", "content": f"다음 텍스트가 부적절한지 판단해주세요: {text}"},
                ]
                
                input_ids = tokenizer.apply_chat_template(chat, return_tensors="pt", tokenize=True)
                input_ids = input_ids.to(device)
                
                output_ids = model.generate(
                    input_ids,
                    max_new_tokens=64,
                    do_sample=True,
                    top_p=0.6,
                    temperature=0.5,
                    repetition_penalty=1.0,
                )
                
                result = tokenizer.batch_decode(output_ids)[0]
                
                # ChatML 형식에서 실제 응답 추출
                response_start = result.rfind("<|im_start|>assistant\n") + len("<|im_start|>assistant\n")
                response_end = result.rfind("<|im_end|>")
                if response_start != -1 and response_end != -1:
                    result = result[response_start:response_end].strip()
                
                logging.info(f"[큐 처리] 검열 결과: {result}\n")
                # TODO: 여기에 백엔드 서버로 결과를 전송하는 코드 추가

                logging.info(f"Moderation result: {result}")
            except Exception as e:
                logging.error(f"[큐 처리] 오류 발생: {str(e)}")
        
        # CPU 100% 방지
        time.sleep(0.01)
    
    logging.info("System Finished")
