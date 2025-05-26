import os
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import Docx2txtLoader

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

CATEGORY_ALIASES = {
    normalize_category("개인정보 노출"): "사칭/사기/개인정보 노출",
    normalize_category("개인 정보 노출"): "사칭/사기/개인정보 노출",
    normalize_category("사칭"): "사칭/사기/개인정보 노출",
    normalize_category("사기"): "사칭/사기/개인정보 노출",
    # 추가 필요시 여기에
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
    except Exception:
        return ""

def validate_spec(response: str) -> bool:
    response = response.strip()
    if response == "검열 불필요: 적절한 표현입니다.":
        return True
    if ": " in response:
        category, reason = response.split(": ", 1)
        if category in CATEGORY_MAPPING.keys() and reason.strip():
            return True
    return False 