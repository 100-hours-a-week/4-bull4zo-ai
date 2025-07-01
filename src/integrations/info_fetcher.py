import os
import requests
import json
from dotenv import load_dotenv
from src.common.logger_config import init_process_logging

# .env 파일에서 환경 변수 로드
load_dotenv()
serper_api_key = os.getenv("SERPER_API_KEY")
logger = init_process_logging("ai")

class InfoFetcher:
    @staticmethod
    def fetch(word: str) -> str:
        logger.info(f"InfoFetcher 시작: word={word}")
        url = "https://google.serper.dev/search"

        payload = json.dumps([{
            "q": word + " 유머",
            "gl": "kr",
            "hl": "ko",
            #"tbs": "qdr:m"
        }])
        headers = {
            'X-API-KEY': serper_api_key,
            'Content-Type': 'application/json'
        }

        try:
            response = requests.request("POST", url, headers=headers, data=payload)
            data = response.json()
            
            # 첫 번째 검색 결과에서 snippet 추출
            if data and len(data) > 0:
                first_result = data[0]
                
                # answerBox에서 snippet이 있는 경우
                if 'answerBox' in first_result and 'snippet' in first_result['answerBox']:
                    return first_result['answerBox']['snippet']
                
                # organic 결과에서 snippet 추출
                if 'organic' in first_result and len(first_result['organic']) > 0:
                    snippets = [item['snippet'] for item in first_result['organic'] if 'snippet' in item][:5] # 상위 5개만 추출
                    return '\n'.join(snippets)
            
            return "검색 결과가 없습니다."
        except Exception as e:
            return f"[ERROR] Serper API 호출 중 오류 발생: {str(e)}"
