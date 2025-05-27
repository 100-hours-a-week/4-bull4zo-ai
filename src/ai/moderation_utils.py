import os
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import Docx2txtLoader
import logging
import datetime
import traceback
import csv

CATEGORY_MAPPING = {
    "욕설/비방": "OFFENSIVE_LANGUAGE",
    "정치": "POLITICAL_CONTENT",
    "음란성/선정성": "SEXUAL_CONTENT",
    "스팸/광고": "SPAM_ADVERTISEMENT",
    "사칭/사기/개인정보 노출": "IMPERSONATION_OR_LEAK", 
    "기타": "OTHER"
}

def normalize_category(cat):
    return cat.replace(" ", "").strip()

# CATEGORY_MAPPING만 추가시, 분류 실패로 인해 아래와 같은 경우 카테고리 매핑 필요
CATEGORY_ALIASES = {
    normalize_category("개인정보 노출"): "사칭/사기/개인정보 노출",
    normalize_category("개인 정보 노출"): "사칭/사기/개인정보 노출",
    normalize_category("사칭"): "사칭/사기/개인정보 노출",
    normalize_category("사기"): "사칭/사기/개인정보 노출",
}

vector_store = None
embeddings = None
text_splitter = None

def get_relevant_context(query: str, k: int = 3, similarity_threshold: float = 0.7) -> str:
    global vector_store
    if vector_store is None:
        return ""
    try:
        results = vector_store.similarity_search_with_score(query, k=k)
        relevant_docs = []
        for doc, score in results:
            similarity = 1 / (1 + score)
            if similarity >= similarity_threshold:
                relevant_docs.append(doc.page_content)
        return "\n\n".join(relevant_docs)
    except Exception as e:
        # error.log에 기록
        logging.basicConfig(filename='logs/ai/error.log', level=logging.ERROR,
                            format='%(asctime)s [%(levelname)s] %(module)s:%(lineno)d - %(message)s')
        logging.error(f"get_relevant_context 예외: {e}\n{traceback.format_exc()}")
        # 날짜별 csv에도 기록
        now = datetime.datetime.now()
        date_str = now.strftime('%Y-%m-%d')
        time_str = now.strftime('%H:%M:%S.%f')[:-3]
        csv_path = f'logs/ai/{date_str}.csv'
        row = [date_str, time_str, 'ERROR', 'moderation', '1', '-', '-', '-', '-', f'get_relevant_context 예외: {e}']
        try:
            with open(csv_path, 'a', encoding='utf-8', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(row)
        except Exception as log_e:
            logging.error(f"CSV 로그 기록 실패: {log_e}")
        return ""

def validate_spec(response: str) -> bool:
    """
    모델 응답이 아래 스펙을 정확히 따르는지 검사:
    - '검열 불필요: 적절한 표현입니다.'
    - '[카테고리]: [사유]' (카테고리는 CATEGORY_MAPPING의 key 중 하나)
    """
    response = response.strip()
    if response == "검열 불필요: 적절한 표현입니다.":
        return True
    if ": " in response:
        category, reason = response.split(": ", 1)
        if category in CATEGORY_MAPPING.keys() and reason.strip():
            return True
    return False 
