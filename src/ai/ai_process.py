import os
import time
import logging
from multiprocessing import Queue
from multiprocessing.synchronize import Event
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer
import torch
from langchain_community.document_loaders import Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

vector_store = None
embeddings = None
text_splitter = None

load_dotenv()
hf_token = os.getenv("HF_TOKEN")
openai_api_key = os.getenv("OPENAI_API_KEY")

if not hf_token:
    raise ValueError("HF_TOKEN is not set. Please check your .env file.")

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
    
    logging.info("Model loaded successfully. Waiting for moderation tasks...")

    logging.info("System Started")

    while not stop_event.is_set():
        if not moderation_queue.empty():
            voteContent = moderation_queue.get()

            try:
                print(f"\n[큐 처리] 검열 요청 처리 시작: {voteContent}")
                
                # RAG를 통해 관련 컨텍스트 검색
                relevant_context = get_relevant_context(voteContent)
                
                # 채팅 형식으로 입력 구성
                chat = [
                    {"role": "system", "content": "당신은 검열 시스템입니다. 입력된 텍스트가 부적절한지 판단해야 합니다. '검열 필요: [이유]' 또는 '검열 불필요: 적절한 표현입니다' 형식으로만 답변하세요."},
                ]
                
                # 관련 컨텍스트가 있다면 추가
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
                
                print(f"[큐 처리] 검열 결과: {result}\n")
                # TODO: 여기에 백엔드 서버로 결과를 전송하는 코드 추가
                
            except Exception as e:
                print(f"[큐 처리] 오류 발생: {str(e)}\n")
        
        # CPU 100% 방지
        time.sleep(0.01)
    
    logging.info("System Finished")
