from typing import Dict, List, Any, Optional
from fastapi import FastAPI, HTTPException, status, Request
from pydantic import BaseModel, Field, field_validator
from enum import Enum
import uvicorn
from queue import Queue
from threading import Thread
import time
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer
import torch
from dotenv import load_dotenv
import os
from langchain_community.document_loaders import Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
import logging
import json
import uuid
from fastapi.responses import JSONResponse
from fastapi.exception_handlers import RequestValidationError
from fastapi.exceptions import RequestValidationError as FastAPIRequestValidationError
from starlette.requests import Request as StarletteRequest
from datetime import datetime, timezone

# API 모델 정의
class ServerStatus(str, Enum):
    RUNNING = "Running"
    IDLE = "Idle"
    ERROR = "Error"

class VoteContent(BaseModel):
    voteContent: str = Field(..., description="검열할 투표 내용")
    
    @field_validator('voteContent')
    @classmethod
    def validate_vote_content(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("voteContent must not be null, empty, or whitespace only.")
        return v.strip()

class APIResponse(BaseModel):
    status: str
    message: str
    data: Optional[Dict[str, Any]] = None

class ServerDetail(BaseModel):
    moderationStatus: str
    voteCreationStatus: str

class AIServerInfo(BaseModel):
    name: str
    status: str
    details: List[ServerDetail]

class StatusResponse(APIResponse):
    data: Dict[str, List[AIServerInfo]]

# FastAPI 앱 생성
app = FastAPI(
    title="HyperCLOVAX 검열 시스템 API",
    description="HyperCLOVAX 기반 검열 시스템 API",
    version="1.0.0"
)

# 전역 변수
model = None
processor = None
tokenizer = None
request_queue = Queue()
vector_store = None
embeddings = None
text_splitter = None

# ========== 로깅 설정 ==========
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("moderation.log", encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ========== JSON 로깅 설정 ==========
class JsonLogFormatter(logging.Formatter):
    def format(self, record):
        log_record = record.msg if isinstance(record.msg, dict) else {"message": record.getMessage()}
        return json.dumps(log_record, ensure_ascii=False)

json_handler = logging.FileHandler("logs/censorship.log", encoding="utf-8")
json_handler.setFormatter(JsonLogFormatter())
json_logger = logging.getLogger("censorship")
json_logger.setLevel(logging.INFO)
json_logger.handlers = []
json_logger.addHandler(json_handler)
json_logger.propagate = False

# ========== 1. 환경 설정 ==========
load_dotenv()
hf_token = os.getenv("HF_TOKEN")
openai_api_key = os.getenv("OPENAI_API_KEY")

# ========== 2. GPU 확인 ==========
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"추론 디바이스: {device}")
if device == "cuda":
    print(f"   GPU: {torch.cuda.get_device_name(0)}\n")

def load_and_process_document(file_path: str) -> None:
    """문서를 로드하고 처리하여 벡터 스토어에 저장합니다."""
    global vector_store
    
    try:
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
        
    except Exception as e:
        print(f"문서 처리 중 오류 발생: {str(e)}")
        raise

def get_relevant_context(query: str, k: int = 3) -> str:
    """쿼리에 관련된 컨텍스트를 검색합니다."""
    if vector_store is None:
        return ""
    
    try:
        results = vector_store.similarity_search(query, k=k)
        return "\n\n".join([doc.page_content for doc in results])
    except Exception as e:
        print(f"컨텍스트 검색 중 오류 발생: {str(e)}")
        return ""

def process_moderation_request():
    while True:
        if not request_queue.empty():
            req_info = request_queue.get()
            try:
                relevant_context = get_relevant_context(req_info["voteContent"])
                chat = [
                    {"role": "system", "content": "당신은 검열 시스템입니다. 입력된 텍스트가 부적절한지 판단해야 합니다. '검열 필요: [이유]' 또는 '검열 불필요: 적절한 표현입니다' 형식으로만 답변하세요."},
                ]
                if relevant_context:
                    chat.append({
                        "role": "system",
                        "content": f"다음은 판단에 참고할 수 있는 관련 문서입니다:\n{relevant_context}"
                    })
                chat.append({
                    "role": "user",
                    "content": f"다음 텍스트가 부적절한지 판단해주세요: {req_info['voteContent']}"
                })
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
                response_start = result.rfind("<|im_start|>assistant\n") + len("<|im_start|>assistant\n")
                response_end = result.rfind("<|im_end|>")
                if response_start != -1 and response_end != -1:
                    parsed_result = result[response_start:response_end].strip()
                else:
                    parsed_result = ""
                if not parsed_result:
                    # 파싱 실패 시 원본 전체를 moderation_result로 남김
                    log_data = {
                        **req_info,
                        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                        "status_code": 200,
                        "moderation_result": result.strip(),
                        "error_message": None
                    }
                else:
                    log_data = {
                        **req_info,
                        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                        "status_code": 200,
                        "moderation_result": parsed_result,
                        "error_message": None
                    }
                json_logger.info(log_data)
            except Exception as e:
                log_data = {
                    **req_info,
                    "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "status_code": 500,
                    "moderation_result": None,
                    "error_message": str(e)
                }
                json_logger.info(log_data)
        time.sleep(0.1)

@app.on_event("startup")
async def startup_event():
    global model, processor, tokenizer, embeddings, text_splitter
    
    try:
        print("모델 로드 중...")
        model_name = "naver-hyperclovax/HyperCLOVAX-SEED-Vision-Instruct-3B"
        
        # 모델 및 토크나이저 로드
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
        
        # RAG 컴포넌트 초기화
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
        
        # 백그라운드 처리 스레드 시작
        background_thread = Thread(target=process_moderation_request, daemon=True)
        background_thread.start()
        
        print("시스템 준비 완료!")
    except Exception as e:
        print(f"초기화 중 오류 발생: {str(e)}")
        model = None

@app.post("/api/v1/moderation", status_code=status.HTTP_202_ACCEPTED, response_model=APIResponse)
async def check_vote_content(vote: VoteContent, request: Request):
    request_id = str(uuid.uuid4())
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    client_ip = request.client.host if request.client else "unknown"
    req_info = {
        "request_id": request_id,
        "timestamp": timestamp,
        "endpoint": "/api/v1/moderation",
        "client_ip": client_ip,
        "voteContent": vote.voteContent
    }
    if not model:
        log_data = {**req_info, "status_code": 500, "moderation_result": None, "error_message": "Moderation system is not initialized"}
        json_logger.info(log_data)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Moderation system is not initialized"
        )
    try:
        # 요청을 큐에 추가 (dict로)
        request_queue.put(req_info)
        log_data = {**req_info, "status_code": 202, "moderation_result": "queued", "error_message": None}
        json_logger.info(log_data)
        return APIResponse(
            status="Accepted",
            message="voteContent has been queued for processing",
            data={"queuePosition": request_queue.qsize()}
        )
    except Exception as e:
        log_data = {**req_info, "status_code": 500, "moderation_result": None, "error_message": str(e)}
        json_logger.info(log_data)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@app.get("/api/v1/status", response_model=StatusResponse)
async def get_status():
    server_status = ServerStatus.RUNNING if model else ServerStatus.ERROR
    moderation_status = ServerStatus.RUNNING if model else ServerStatus.ERROR
    
    return StatusResponse(
        status="OK",
        message="",
        data={
            "aiServerStatus": [
                {
                    "name": "HyperCLOVAX Server #1",
                    "status": server_status,
                    "details": [
                        {
                            "moderationStatus": moderation_status,
                            "voteCreationStatus": ServerStatus.IDLE
                        }
                    ]
                }
            ]
        }
    )

# Pydantic validation error(400) 핸들러 추가
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: StarletteRequest, exc: FastAPIRequestValidationError):
    request_id = str(uuid.uuid4())
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    client_ip = request.client.host if request.client else "unknown"
    input_text = None
    try:
        body = await request.json()
        input_text = body.get("voteContent")
    except Exception:
        pass
    log_data = {
        "request_id": request_id,
        "timestamp": timestamp,
        "endpoint": "/api/v1/moderation",
        "client_ip": client_ip,
        "voteContent": input_text,
        "status_code": 400,
        "moderation_result": None,
        "error_message": "voteContent must not be null, empty, or whitespace only."
    }
    json_logger.info(log_data)
    return JSONResponse(
        status_code=400,
        content={
            "status": "Bad Request",
            "message": "voteContent must not be null, empty, or whitespace only.",
            "data": None
        }
    )

if __name__ == "__main__":
    uvicorn.run("clova_server:app", host="0.0.0.0", port=8000, reload=False)
