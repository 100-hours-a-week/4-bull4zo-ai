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

# 한글 카테고리를 영어 ENUM으로 맵핑
CATEGORY_MAPPING = {
    "욕설/비방": "OFFENSIVE_LANGUAGE",
    "정치": "POLITICAL_CONTENT",
    "음란성/선정성": "SEXUAL_CONTENT",
    "스팸/광고": "SPAM_ADVERTISEMENT",
    "사칭/사기/개인정보 노출": "IMPERSONATION_OR_LEAK", 
    "기타": "OTHER"
}

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

def run_model_process(stop_event: Event, moderation_queue: Queue):
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
        if not moderation_queue.empty():
            moderation_request: ModerationRequest = moderation_queue.get()
            content = moderation_request.content
            request_id = str(moderation_request.voteId)

            try:
                logger.info(f"검열 요청 처리 시작", 
                           extra={
                               "section": "moderation", 
                               "request_id": request_id,
                               "content": content
                            })
                
                # RAG를 통해 관련 컨텍스트 검색
                relevant_context = get_relevant_context(content)
                
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
                    "content": f"다음 텍스트가 부적절한지 판단하고 카테고리를 분류해주세요: {content}"
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
                logger.info(f"추론 시간: {inference_time:.2f}초", 
                           extra={"section": "moderation", "request_id": request_id})
                
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
                    logger.error(f"응답 파싱 중 오류: {str(e)}", exc_info=True,
                                extra={"section": "moderation", "request_id": request_id})
                    result = result.strip()
                
                if not result:
                    logger.warning("모델 응답이 비어있습니다.",
                                 extra={"section": "moderation", "request_id": request_id})
                
                # 모델 응답 로깅
                logger.info(f"모델 응답: '{result}'", 
                           extra={
                               "section": "moderation",
                               "request_id": request_id,
                               "model_version": "v1.0.0",
                               "inference_time": f"{inference_time:.2f}s"
                           })
                
                # 결과 처리
                final_result = ""
                final_reason = ""
                final_reason_detail = ""
                
                if result:  # 결과가 비어있지 않은 경우에만 처리
                    if "검열 불필요" in result:
                        final_result = "APPROVED"
                        final_reason = "적절한 표현입니다"
                        pred_label = "APPROVED"
                        pred_score = "1.0"
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
                        pred_label = found_category_en
                        pred_score = "0.9"  # 예시 점수
                else:
                    final_result = "ERROR"
                    final_reason = "OTHER"
                    final_reason_detail = "검열 결과를 얻을 수 없습니다"
                    pred_label = "ERROR"
                    pred_score = "0.0"
                    logger.error("모델 응답이 없어 검열할 수 없습니다.",
                                extra={"section": "moderation", "request_id": request_id})
                
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
                
                # 검열 결과 로깅
                logger.info(f"검열 결과: {final_result}, 카테고리={final_reason}, 이유='{final_reason_detail}'",
                           extra={
                               "section": "server", 
                               "request_id": request_id,
                               "pred_label": pred_label,
                               "pred_score": pred_score,
                               "model_version": "v1.0.0"
                           })

                # 백엔드 서버로 검열 결과 전송
                try:
                    # voteId를 문자열로 변환
                    request_id = str(moderation_request.voteId)
                    
                    logger.info(f"검열 결과 전송 시작", 
                               extra={"section": "server", "request_id": request_id})
                    
                    # 실제 콜백 요청 보내기
                    response = requests.post(callback_url, json=moderation_result_request.dict(), headers=headers)
                    
                    # 응답 결과 로깅
                    if response.status_code == 201:
                        logger.info(f"검열 결과 전송 성공: HTTP {response.status_code}",
                                   extra={"section": "server", "request_id": request_id})
                        
                    elif response.status_code == 400:
                        logger.error(f"검열 결과 전송 실패: HTTP {response.status_code}",
                                    extra={"section": "server", "request_id": request_id})
                    elif response.status_code == 404:
                        logger.error(f"검열 결과 전송 실패: HTTP {response.status_code}",
                                    extra={"section": "server", "request_id": request_id})
                    elif response.status_code == 500:
                        logger.error(f"검열 결과 전송 실패: HTTP {response.status_code}",
                                    extra={"section": "server", "request_id": request_id})
                    else:
                        logger.error(f"검열 결과 전송 실패: HTTP {response.status_code}",
                                    extra={"section": "server", "request_id": request_id})
                except requests.exceptions.RequestException as e:
                    error_msg = f"검열 결과 전송 중 네트워크 오류: {str(e)}"
                    logger.error(error_msg, exc_info=True,
                                extra={"section": "server", "request_id": request_id})
                
            except Exception as e:
                error_msg = f"검열 처리 중 오류 발생: {str(e)}"
                logger.error(error_msg, exc_info=True,
                            extra={"section": "moderation", "request_id": request_id})
        
        # CPU 100% 방지
        time.sleep(0.01)
    
    logger.info("System Finished",
               extra={"section": "system", "request_id": "shutdown"})
    
    # 로깅 시스템 종료
    shutdown_logging("ai")
