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

vector_store = None
embeddings = None
text_splitter = None

load_dotenv()
hf_token = os.getenv("HF_TOKEN")
openai_api_key = os.getenv("OPENAI_API_KEY")
be_server_ip = os.getenv("BE_SERVER_IP")
be_server_port = os.getenv("BE_SERVER_PORT")

callback_url = f"http://{be_server_ip}:{be_server_port}/api/v1/ai/votes/moderation/callback"

if not hf_token:
    raise ValueError("HF_TOKEN is not set. Please check your .env file.")

# 기존 로거 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler()  # 콘솔 출력
    ]
)

# RAG 전용 로거 설정
rag_logger = logging.getLogger('rag')
rag_logger.setLevel(logging.INFO)
rag_logger.propagate = False  # 부모 로거로 전파하지 않음
rag_handler = logging.FileHandler('logs/rag.log')
rag_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
rag_logger.addHandler(rag_handler)

def load_and_process_document(file_path: str) -> None:
    """문서를 로드하고 처리하여 벡터 스토어에 저장합니다."""
    global vector_store, text_splitter
    
    try:
        if text_splitter is None:
            raise ValueError("text_splitter가 초기화되지 않았습니다. RAG 컴포넌트가 먼저 초기화되어야 합니다.")
            
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
        rag_logger.info(f"문서 '{file_path}' 처리 완료")
        
    except Exception as e:
        rag_logger.error(f"문서 처리 중 오류 발생: {str(e)}")
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
            logging.info("[RAG] 관련 문서를 찾았습니다")
        else:
            logging.info("[RAG] 관련 문서가 없습니다")
            
        return "\n\n".join(relevant_docs)
    except Exception as e:
        logging.error(f"[RAG] 오류 발생: {str(e)}")
        return ""

def run_model_process(stop_event: Event, moderation_queue: Queue):
    # Device 설정
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"추론 디바이스: {device}")
    if device == "cuda":
        logging.info(f"   GPU: {torch.cuda.get_device_name(0)}\n")

    # LLM 로딩
    logging.info("Loading model...")

    model_name = "naver-hyperclovax/HyperCLOVAX-SEED-Vision-Instruct-3B"

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
    
    logging.info("Model loaded successfully. Waiting for moderation tasks...")

    logging.info("System Started")

    while not stop_event.is_set():
        if not moderation_queue.empty():
            moderation_request: ModerationRequest = moderation_queue.get()
            voteContent = moderation_request.voteContent

            try:
                logging.info(f"\n[큐 처리] 검열 요청 처리 시작: {voteContent}")
                
                # 검열 입력만 로그
                logging.info(f"[검열 입력] {voteContent}")
                
                # RAG를 통해 관련 컨텍스트 검색
                relevant_context = get_relevant_context(voteContent)
                
                # 채팅 형식으로 입력 구성
                chat = [
                    {"role": "system", "content": "당신은 검열 시스템입니다. 입력된 텍스트가 부적절한지 간단히 판단해야 합니다. 긴 설명은 하지 말고, '검열 필요: [이유]' 또는 '검열 불필요: 적절한 표현입니다' 형식으로만 답변하세요."},
                ]
                
                # 관련 컨텍스트가 있다면 추가 (로그에는 출력하지 않음)
                if relevant_context:
                    chat.append({
                        "role": "system",
                        "content": f"다음은 판단에 참고할 수 있는 관련 문서입니다:\n{relevant_context}"
                    })
                
                chat.append({
                    "role": "user",
                    "content": f"다음 텍스트가 부적절한지 판단해주세요: {voteContent}"
                })
                
                input_ids = tokenizer.apply_chat_template(chat, return_tensors="pt", tokenize=True)
                input_ids = input_ids.to(device)
                
                output_ids = model.generate(
                    input_ids,
                    max_new_tokens=128,  # 토큰 수 증가
                    do_sample=True,
                    top_p=0.9,          # 더 결정적인 응답을 위해 조정
                    temperature=0.3,     # 더 결정적인 응답을 위해 조정
                    repetition_penalty=1.2,  # 반복 방지
                    pad_token_id=tokenizer.pad_token_id,  # 패딩 토큰 지정
                    eos_token_id=tokenizer.eos_token_id,  # 종료 토큰 지정
                )
                
                result = tokenizer.batch_decode(output_ids)[0]
                
                # ChatML 형식에서 실제 응답 추출 (수정된 로직)
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
                    result = result.strip()
                
                if not result:
                    logging.warning("[검열 출력] 최종 응답이 비어있습니다.")
                
                # 검열 출력만 로그
                logging.info(f"[검열 출력] {result}")
                
                final_result = ""
                final_reason = ""
                
                if result:  # 결과가 비어있지 않은 경우에만 처리
                    if "검열 불필요" in result:
                        final_result = "APPROVED"
                        final_reason = "적절한 표현입니다"
                    else:
                        final_result = "REJECTED"
                        # 검열 필요: [이유] 형식에서 이유 부분만 추출
                        if ": " in result:
                            final_reason = result.split(": ", 1)[1]
                        else:
                            final_reason = result
                else:
                    final_result = "ERROR"
                    final_reason = "검열 결과를 얻을 수 없습니다"
                
                # TODO: AI Server Version 규격 정의 및 설정 필요
                version = "1.0.0"  # 임시 버전 설정

                headers = {
                    "Content-Type": "application/json"
                }
                
                # TODO: 형식 관련 논의 필요 (w/ BE)
                moderation_result_request = ModerationResultRequest(
                    voteId=moderation_request.voteId if moderation_request.voteId is not None else 0,
                    result=final_result,
                    reason=final_reason, # TODO: 검열 카테고리 필요
                    reasonDetail=final_reason,
                    version=version
                )

                # TODO: 로깅 필요
                # 콜백 전송 대신 결과 출력
                print("\n=== 검열 결과 ===")
                print(f"Vote ID: {moderation_result_request.voteId}")
                print(f"Result: {moderation_result_request.result}")
                print(f"Reason: {moderation_result_request.reason}")
                print(f"Version: {moderation_result_request.version}")
                print("================\n")

                try:
                    response = requests.post(callback_url, json=moderation_result_request.dict(), headers=headers)
                    if response.status_code == 201:
                        result = response.json()
                        logging.info("[201 Created] 저장 성공:", result)
                    elif response.status_code == 400:
                        logging.error("[400 Bad Request] 요청이 잘못되었습니다:", response.text)
                    elif response.status_code == 404:
                        logging.error("[404 Not Found] 경로를 찾을 수 없습니다:", response.text)
                    elif response.status_code == 500:
                        logging.error("[500 Internal Server Error] 서버 오류:", response.text)
                    else:
                        logging.error(f"[{response.status_code}] 예상치 못한 응답:", response.text)
                except requests.exceptions.RequestException as e:
                    logging.error("⚠️ 요청 중 예외 발생:", e)
                
            except Exception as e:
                # TODO: Retry, 등 검열 실패 시 추가 동작에 관해 처리 필요
                logging.info(f"[큐 처리] 오류 발생: {str(e)}\n")
        
        # CPU 100% 방지
        time.sleep(0.01)
    
    logging.info("System Finished")
