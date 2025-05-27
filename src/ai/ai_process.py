import os
import time
from multiprocessing import Queue
from multiprocessing.synchronize import Event
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer
import torch
from langchain_community.document_loaders import Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from src.api.dtos.moderation_request import ModerationRequest
from src.api.dtos.moderation_result_request import ModerationResultRequest
import requests
import json
import datetime
from src.common.logger_config import init_process_logging, shutdown_logging
from src.ai.vote_generator import VoteGenerator
from src.ai.moderation_llm import moderation_pipeline
from src.ai.moderation_utils import get_relevant_context, validate_spec, CATEGORY_MAPPING, normalize_category
from src.integrations.info_fetcher import InfoFetcher
from src.integrations.moderation import moderate
from src.integrations.delivery import Delivery

# 로거 초기화
logger = init_process_logging("ai")

vector_store = None
embeddings = None
text_splitter = None

load_dotenv()
hf_token = os.getenv("HF_TOKEN")
openai_api_key = os.getenv("OPENAI_API_KEY")
environment = os.getenv("ENVIRONMENT").lower()
be_server_ip = "127.0.0.1" if environment == "dev" else os.getenv("BE_SERVER_IP")
be_server_port = os.getenv("BE_SERVER_PORT")

callback_url = f"http://{be_server_ip}:{be_server_port}/api/v1/ai/votes/moderation/callback"

if not hf_token:
    logger.error("HF_TOKEN is not set. Please check your .env file.", 
                extra={"section": "system", "request_id": "init"})
    raise ValueError("HF_TOKEN is not set. Please check your .env file.")

def load_and_process_document(file_path: str) -> None:
    """문서를 로드하고 처리하여 벡터 스토어에 저장합니다."""
    global vector_store, text_splitter
    
    try:
        if text_splitter is None:
            error_msg = "text_splitter가 초기화되지 않았습니다. RAG 컴포넌트가 먼저 초기화되어야 합니다."
            logger.error(error_msg, 
                        extra={"section": "rag", "request_id": "init"})
            raise ValueError(error_msg)
            
        # 문서 로드
        loader = Docx2txtLoader(file_path)
        documents = loader.load()
        
        # 문서를 청크로 분할
        splits = text_splitter.split_documents(documents)
        
        # Chroma 벡터 스토어 생성 또는 업데이트
        if vector_store is None:
            vector_store = Chroma.from_documents(
                documents=splits,
                embedding=embeddings,
                persist_directory="./chroma_db"
            )
        else:
            vector_store.add_documents(splits)
        
        # 변경사항 저장
        vector_store.persist()
        logger.info(f"문서 '{file_path}' 처리 완료", 
                   extra={"section": "rag", "request_id": "init"})
        
    except Exception as e:
        error_msg = f"문서 처리 중 오류 발생: {str(e)}"
        logger.error(error_msg, exc_info=True,
                    extra={"section": "rag", "request_id": "init"})
        raise

def get_relevant_context(query: str, k: int = 3, similarity_threshold: float = 0.7) -> str:
    """쿼리에 관련된 컨텍스트를 검색합니다."""
    if vector_store is None:
        return ""
    
    try:
        # similarity_search_with_score를 사용하여 유사도 점수도 함께 받음
        results = vector_store.similarity_search_with_score(query, k=k)
        
        # 유사도가 threshold를 넘는 문서만 선택
        relevant_docs = []
        for doc, score in results:
            similarity = 1 / (1 + score)  # 거리를 0-1 사이의 유사도 점수로 변환
            if similarity >= similarity_threshold:
                relevant_docs.append(doc.page_content)
        
        if relevant_docs:
            logger.info("[RAG] 관련 문서를 찾았습니다", 
                       extra={"section": "rag"})
        else:
            logger.info("[RAG] 관련 문서가 없습니다", 
                       extra={"section": "rag"})
            
        return "\n\n".join(relevant_docs)
    except Exception as e:
        error_msg = f"[RAG] 오류 발생: {str(e)}"
        logger.error(error_msg, exc_info=True,
                    extra={"section": "rag"})
        return ""

def run_model_process(stop_event: Event, moderation_task_queue: Queue, result_queue: Queue):
    # Device 설정
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"추론 디바이스: {device}", 
               extra={"section": "system", "request_id": "init"})
    if device == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}", 
                   extra={"section": "system", "request_id": "init"})

    # LLM 로딩
    logger.info("Loading model...", 
               extra={"section": "system", "request_id": "init"})

    model_name = "naver-hyperclovax/HyperCLOVAX-SEED-Vision-Instruct-3B"

    try:
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
    except Exception as e:
        error_msg = f"모델 로딩 중 오류 발생: {str(e)}"
        logger.error(error_msg, exc_info=True, 
                    extra={"section": "system", "request_id": "init"})
        raise

    # RAG 컴포넌트 초기화
    global embeddings, text_splitter, vector_store
    
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    
    # 기존 문서 로드
    docx_path = "RAG_data.docx"
    if os.path.exists(docx_path):
        load_and_process_document(docx_path)
    
    logger.info("Model loaded successfully. Waiting for moderation tasks...", 
               extra={"section": "system", "request_id": "init"})

    logger.info("System Started", 
               extra={"section": "system", "request_id": "init"})

    while not stop_event.is_set():
        if not moderation_task_queue.empty():
            task = moderation_task_queue.get()
            if isinstance(task, dict) and "type" in task:
                task_type = task["type"]
                data = task["data"]
                if task_type == "moderation":
                    result = moderation_pipeline(data, model, tokenizer, device, callback_url, logger)
                    result_queue.put(result)
                elif task_type == "vote":
                    word_id = data.get("word_id")
                    word = data.get("word", "")
                    info = InfoFetcher.fetch(word)
                    vote = VoteGenerator().generate(word, info, model, tokenizer)
                    moderation_request = ModerationRequest(content=vote["content"], voteId=word_id)
                    mod = moderate(moderation_request, model, tokenizer, device, callback_url, logger)
                    if mod["result"] == "REJECTED":
                        # ModerationLog 저장
                        logger.info(
                            json.dumps({
                                "vote_id": word_id,
                                "content": vote["content"],
                                "model_response": mod.get("model_response", ""),
                                "inference_time": mod.get("inference_time", "")
                            }),
                            extra={"section": "moderation", "request_id": str(word_id)}
                        )
                        result_queue.put({"word_id": word_id, "status": "rejected"})
                    else:
                        Delivery.push(word_id, vote, logger, str(word_id))
                        result_queue.put({"word_id": word_id, "status": "delivered"})
            else:
                result = moderation_pipeline(task, model, tokenizer, device, callback_url, logger)
                result_queue.put(result)
        time.sleep(0.01)
    
    logger.info("System Finished",
               extra={"section": "system", "request_id": "shutdown"})
    
    # 로깅 시스템 종료
    shutdown_logging("ai")

