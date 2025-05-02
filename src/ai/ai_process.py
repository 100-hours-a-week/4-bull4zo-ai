import os
import time
import logging
from multiprocessing import Queue
from multiprocessing.synchronize import Event
from dotenv import load_dotenv
from fastapi import requests
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer
import torch
from langchain_community.document_loaders import Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from src.api.dtos.moderation_request import ModerationRequest
from src.api.dtos.moderation_result_request import ModerationResultRequest
import requests
import logging.handlers
import csv
import datetime
import json

# 한글 카테고리를 영어 ENUM으로 맵핑
CATEGORY_MAPPING = {
    "욕설/비방": "OFFENSIVE_LANGUAGE",
    "정치": "POLITICAL_CONTENT",
    "음란성/선정성": "SEXUAL_CONTENT",
    "스팸/광고": "SPAM_ADVERTISEMENT",
    "사칭/사기/개인정보 노출": "IMPERSONATION_OR_LEAK", 
    "기타": "OTHER"
}

# 로그 디렉토리 생성
def ensure_log_dirs():
    """로그 디렉토리가 존재하는지 확인하고 없으면 생성합니다."""
    dirs = [
        "logs/ai",
        "logs/api"
    ]
    for dir_path in dirs:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

# 로그 디렉토리 생성
ensure_log_dirs()

# 현재 날짜 기반 파일명
today = datetime.datetime.now().strftime("%Y-%m-%d")
csv_log_file = f"logs/ai/{today}.csv"

# CSV 로거 설정을 위한 커스텀 핸들러
class CSVFileHandler(logging.FileHandler):
    def __init__(self, filename, mode='a', encoding=None, delay=False):
        logging.FileHandler.__init__(self, filename, mode, encoding, delay)
        self.header_written = os.path.exists(filename) and os.path.getsize(filename) > 0

    def emit(self, record):
        if self.stream is None:
            self.stream = self._open()
        
        # CSV 헤더가 필요한 경우
        if not self.header_written:
            writer = csv.writer(self.stream)
            writer.writerow(['timestamp', 'level', 'message'])
            self.header_written = True
        
        # CSV 행 추가
        if record.levelno >= self.level:
            writer = csv.writer(self.stream)
            timestamp = datetime.datetime.fromtimestamp(record.created).strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
            writer.writerow([timestamp, record.levelname, record.getMessage()])
            self.flush()

# 기본 로거 설정
logger = logging.getLogger('ai')
logger.setLevel(logging.INFO)
logger.propagate = False  # 부모 로거로 전파 방지

# 1. CSV 파일 핸들러 (날짜별 서비스 흐름 로깅)
csv_handler = CSVFileHandler(csv_log_file)
csv_formatter = logging.Formatter('%(asctime)s,%(levelname)s,%(message)s', '%Y-%m-%d %H:%M:%S')
csv_handler.setLevel(logging.INFO)
csv_handler.setFormatter(csv_formatter)
logger.addHandler(csv_handler)

# 2. 검열 세부 로그 (모델 개선용 누적 데이터)
moderation_logger = logging.getLogger('ai.moderation')
moderation_logger.setLevel(logging.INFO)
moderation_logger.propagate = False

moderation_handler = logging.FileHandler('logs/ai/moderation.log')
moderation_formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
moderation_handler.setFormatter(moderation_formatter)
moderation_logger.addHandler(moderation_handler)

# 3. 에러 로그 (장애 추적)
error_logger = logging.getLogger('ai.error')
error_logger.setLevel(logging.ERROR)
error_logger.propagate = False

error_handler = logging.FileHandler('logs/ai/error.log')
error_formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
error_handler.setFormatter(error_formatter)
error_logger.addHandler(error_handler)

# 콘솔 출력 핸들러 (디버깅용)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
console_handler.setFormatter(console_formatter)
logger.addHandler(console_handler)

# JSON 로그 포맷터 (모델 개선용)
class JsonFormatter(logging.Formatter):
    def format(self, record):
        if isinstance(record.msg, dict):
            return json.dumps(record.msg, ensure_ascii=False)
        return json.dumps({
            'timestamp': datetime.datetime.fromtimestamp(record.created).strftime('%Y-%m-%d %H:%M:%S.%f')[:-3],
            'level': record.levelname,
            'message': record.getMessage()
        }, ensure_ascii=False)

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
    error_logger.error("HF_TOKEN is not set. Please check your .env file.")
    raise ValueError("HF_TOKEN is not set. Please check your .env file.")

def load_and_process_document(file_path: str) -> None:
    """문서를 로드하고 처리하여 벡터 스토어에 저장합니다."""
    global vector_store, text_splitter
    
    try:
        if text_splitter is None:
            error_msg = "text_splitter가 초기화되지 않았습니다. RAG 컴포넌트가 먼저 초기화되어야 합니다."
            error_logger.error(error_msg)
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
        logger.info(f"문서 '{file_path}' 처리 완료")
        
    except Exception as e:
        error_msg = f"문서 처리 중 오류 발생: {str(e)}"
        error_logger.error(error_msg, exc_info=True)
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
            logger.info("[RAG] 관련 문서를 찾았습니다")
        else:
            logger.info("[RAG] 관련 문서가 없습니다")
            
        return "\n\n".join(relevant_docs)
    except Exception as e:
        error_msg = f"[RAG] 오류 발생: {str(e)}"
        error_logger.error(error_msg, exc_info=True)
        return ""

def run_model_process(stop_event: Event, moderation_queue: Queue):
    # Device 설정
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"추론 디바이스: {device}")
    if device == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

    # LLM 로딩
    logger.info("Loading model...")

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
        error_logger.error(error_msg, exc_info=True)
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
    
    logger.info("Model loaded successfully. Waiting for moderation tasks...")

    logger.info("System Started")

    while not stop_event.is_set():
        if not moderation_queue.empty():
            moderation_request: ModerationRequest = moderation_queue.get()
            voteContent = moderation_request.voteContent

            try:
                logger.info(f"검열 요청 처리 시작: {voteContent}")
                
                # RAG를 통해 관련 컨텍스트 검색
                relevant_context = get_relevant_context(voteContent)
                
                # 채팅 형식으로 입력 구성
                chat = [
                    {"role": "system", "content": "당신은 검열 시스템입니다. 입력된 텍스트가 부적절한지 판단하고 분류해야 합니다. 반드시 다음 형식으로만 답변하세요: 부적절한 내용이면 '검열 필요: [카테고리] [이유]', 적절한 내용이면 '검열 불필요: 적절한 표현입니다'. 카테고리는 다음 중 하나여야 합니다: '욕설/비방', '정치', '음란성/선정성', '스팸/광고', '사칭/사기/개인정보 노출', '기타'. 욕설의 경우 초성만 사용하거나 일부분만 사용한 경우('ㅅㅂ', '씨1발', '개ㅅㅂ' 등)도 욕설/비방으로 분류해야 합니다."},
                ]
                
                # 관련 컨텍스트가 있다면 추가
                if relevant_context:
                    chat.append({
                        "role": "system",
                        "content": f"다음은 판단에 참고할 수 있는 관련 문서입니다:\n{relevant_context}"
                    })
                
                chat.append({
                    "role": "user",
                    "content": f"다음 텍스트가 부적절한지 판단하고 카테고리를 분류해주세요: {voteContent}"
                })
                
                start_time = time.time()
                
                input_ids = tokenizer.apply_chat_template(chat, return_tensors="pt", tokenize=True)
                input_ids = input_ids.to(device)
                
                output_ids = model.generate(
                    input_ids,
                    max_new_tokens=128,
                    do_sample=True,
                    top_p=0.9,
                    temperature=0.3,
                    repetition_penalty=1.2,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
                
                inference_time = time.time() - start_time
                logger.info(f"추론 시간: {inference_time:.2f}초")
                
                result = tokenizer.batch_decode(output_ids)[0]
                
                # ChatML 형식에서 실제 응답 추출
                try:
                    # 마지막 assistant 응답 찾기
                    start_marker = "<|im_start|>assistant\n"
                    end_marker = "<|im_end|>"
                    
                    # 모든 assistant 응답의 시작 위치 찾기
                    start_positions = []
                    pos = 0
                    while True:
                        pos = result.find(start_marker, pos)
                        if pos == -1:
                            break
                        start_positions.append(pos)
                        pos += 1
                    
                    if start_positions:
                        # 마지막 assistant 응답 위치
                        last_start = start_positions[-1] + len(start_marker)
                        # 마지막 응답 이후의 첫 번째 end_marker
                        last_end = result.find(end_marker, last_start)
                        
                        if last_end != -1:
                            result = result[last_start:last_end].strip()
                        else:
                            result = result[last_start:].strip()
                    else:
                        result = result.strip()
                except Exception as e:
                    error_logger.error(f"응답 파싱 중 오류: {str(e)}", exc_info=True)
                    result = result.strip()
                
                if not result:
                    error_logger.warning("모델 응답이 비어있습니다.")
                
                # 검열 출력을 moderation.log에 저장
                moderation_log_data = {
                    "vote_id": moderation_request.voteId,
                    "content": voteContent,
                    "model_response": result,
                    "inference_time": f"{inference_time:.2f}s"
                }
                moderation_logger.info(json.dumps(moderation_log_data, ensure_ascii=False))
                
                # 동일한 정보를 CSV 로그에도 기록 (요약 형태로)
                logger.info(f"모델 응답: '{result}'")
                
                # 결과 처리
                final_result = ""
                final_reason = ""
                final_reason_detail = ""
                
                if result:  # 결과가 비어있지 않은 경우에만 처리
                    if "검열 불필요" in result:
                        final_result = "APPROVED"
                        final_reason = "적절한 표현입니다"
                    else:
                        final_result = "REJECTED"
                        
                        # 카테고리 식별 (한국어)
                        kr_categories = list(CATEGORY_MAPPING.keys())
                        found_category_kr = "기타"  # 기본값 (한국어)
                        reason_detail = ""
                        
                        # "검열 필요: [카테고리] [이유]" 형식 처리
                        if ": " in result:
                            content_part = result.split(": ", 1)[1]
                            
                            for category in kr_categories:
                                if category in content_part:
                                    found_category_kr = category
                                    reason_detail = content_part.replace(category, "", 1).strip()
                                    break
                            
                            # 이유가 없을 경우 기본 메시지 설정
                            if not reason_detail:
                                reason_detail = "부적절한 내용이 감지되었습니다."
                        else:
                            # "검열 필요:" 없이 카테고리만 바로 있는 경우 (예: "욕설/비방")
                            for category in kr_categories:
                                if category in result:
                                    found_category_kr = category
                                    break
                            
                            # 카테고리 외의 내용은 상세 이유로
                            for category in kr_categories:
                                result = result.replace(category, "", 1)
                            
                            reason_detail = result.strip()
                            if not reason_detail:
                                reason_detail = "부적절한 내용이 감지되었습니다."
                        
                        # 한국어 카테고리를 영어 ENUM 코드로 변환
                        found_category_en = CATEGORY_MAPPING.get(found_category_kr, "OTHER")
                        
                        final_reason = found_category_en  # 영어 ENUM 코드 사용
                        final_reason_detail = reason_detail
                else:
                    final_result = "ERROR"
                    final_reason = "OTHER"
                    final_reason_detail = "검열 결과를 얻을 수 없습니다"
                    error_logger.error("모델 응답이 없어 검열할 수 없습니다.")
                
                version = "1.0.0"  # 버전 정보

                headers = {
                    "Content-Type": "application/json"
                }
                
                moderation_result_request = ModerationResultRequest(
                    voteId=moderation_request.voteId if moderation_request.voteId is not None else 0,
                    result=final_result,
                    reason=final_reason,
                    reasonDetail=final_reason_detail,
                    version=version
                )
                
                # CSV 로그에 검열 결과 요약 기록
                logger.info(f"검열 결과: ID={moderation_request.voteId}, 결과={final_result}, 카테고리={final_reason}, 이유='{final_reason_detail}'")

                try:
                    response = requests.post(callback_url, json=moderation_result_request.dict(), headers=headers)
                    if response.status_code == 201:
                        result = response.json()
                        logger.info(f"검열 결과 전송 성공: HTTP {response.status_code}")
                    elif response.status_code == 400:
                        error_msg = f"검열 결과 전송 실패 [400]: {response.text}"
                        error_logger.error(error_msg)
                    elif response.status_code == 404:
                        error_msg = f"검열 결과 전송 실패 [404]: {response.text}"
                        error_logger.error(error_msg)
                    elif response.status_code == 500:
                        error_msg = f"검열 결과 전송 실패 [500]: {response.text}"
                        error_logger.error(error_msg)
                    else:
                        error_msg = f"검열 결과 전송 실패 [{response.status_code}]: {response.text}"
                        error_logger.error(error_msg)
                except requests.exceptions.RequestException as e:
                    error_msg = f"검열 결과 전송 중 네트워크 오류: {str(e)}"
                    error_logger.error(error_msg, exc_info=True)
                
            except Exception as e:
                error_msg = f"검열 처리 중 오류 발생: {str(e)}"
                error_logger.error(error_msg, exc_info=True)
        
        # CPU 100% 방지
        time.sleep(0.01)
    
    logger.info("System Finished")
