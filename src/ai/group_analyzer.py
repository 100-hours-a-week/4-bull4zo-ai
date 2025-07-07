import json
import os
import requests
from src.ai.moderation_llm import parse_moderation_response
from src.ai.model_manager import model, tokenizer
from src.version import __version__ as MODEL_VERSION
from src.common.logger_config import init_process_logging
import re
import pandas as pd
from sqlalchemy import create_engine

DB_USER = str(os.getenv("DB_USER"))
DB_PASS = str(os.getenv("DB_PASS"))
DB_HOST = str(os.getenv("DB_HOST"))
DB_PORT = int(os.getenv("DB_PORT"))
DB_NAME = str(os.getenv("DB_NAME"))
BE_SERVER_IP = os.getenv("BE_SERVER_IP")
BE_SERVER_PORT = os.getenv("BE_SERVER_PORT")

class GroupAnalyzer:
    def _get_group_data(self, start_date, end_date, logger):
        logger.info(f"데이터 조회 시작")

        # DB 연결 정보 (from .env)
        user = DB_USER
        password = DB_PASS
        host = DB_HOST
        port = DB_PORT
        db = DB_NAME

        # SQLAlchemy 엔진 생성 (MySQL + pymysql)
        engine = create_engine(f"mysql+pymysql://{user}:{password}@{host}:{port}/{db}")

        # SQL 쿼리
        query = f"""
        SELECT
            g.id `group_id`,
            g.name `group_name`,
            v.content `vote`,
            MAX(CASE WHEN vr.option_number = 1 THEN vr.count END) AS `yes`,
            MAX(CASE WHEN vr.option_number = 2 THEN vr.count END) AS `no`,
            GROUP_CONCAT(DISTINCT c.content) AS `comment`
        FROM
            `vote` v
        LEFT JOIN
            `vote_result` vr
            ON v.id = vr.vote_id
        LEFT JOIN
            `comment` c
            ON v.id = c.vote_id
        LEFT JOIN
            `group` g
            ON v.group_id = g.id
        WHERE
            g.name != '공개'
            AND g.name NOT LIKE '%%테스트%%'
            AND v.vote_status IN ('OPEN', 'CLOSED')
            AND v.created_at BETWEEN '{start_date}' AND '{end_date}'
        GROUP BY
            g.id,
            g.name,
            v.id"""

        # pandas로 SQL 실행 결과 바로 읽기
        try:
            group_data_df = pd.read_sql(query, con=engine)
            group_data_df['yes'] = group_data_df['yes'].fillna(0)
            group_data_df['no'] = group_data_df['no'].fillna(0)
            
            logger.info("데이터 조회 결과")
            logger.info(group_data_df)
        except Exception as e:
            logger.error(f"[Error] Error is occurred: {e}")
            logger.error(group_data_df)
            return None

        return group_data_df

    def _generate_analysis(self, prompt: str, model, tokenizer, device) -> str:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=1024,
            do_sample=False,
            repetition_penalty=1.2,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id)
        return tokenizer.batch_decode(outputs)[0]

    def _analyze_group(self, group, model, tokenizer, device):
        prompt = """
######################  SYSTEM  ######################
당신은 투표 데이터 분석가입니다.
반드시 [FORMAT(JSON)] 형식으로 [OUTPUT] 이후에 한 번만 출력해야 합니다.
출력 형식 이외의 지시문, Prompt, Template, 그리고 어떠한 다른 Instructions도 출력하지 마세요.
######################################################

#######################  USER  #######################
[DATA]
""" + f"{group}" + """
[/DATA]

[FORMAT(JSON)]
{
    "group_id": {group_id}
    "group_name": {group_name}
    "vote_summary": {vote_summary},
    "comment_summary": {comment_summary},
    "emotion": {[emotion, emotion, emotion]},
    "top_keywords": {[keyword, keyword, keyword]},
    "modelReview": {[review_line, review_line, review_line]}
}

[INFORMATION]
- {vote_summary}는 입력 데이터의 vote들을 분석하여 요약 정리한 내용을 출력할 것.
- {comment_summary}는 입력 데이터의 comment들을 분석하여 요약 정리한 내용을 출력할 것.
- {[emotion, emotion, ...]}은 투표 및 댓글을 기반으로 감정이 어떤지, 대표 감정 3가지만 출력할 것.
- {[keyword, keyword, keyword]}는 투표 및 댓글을 기반으로 대표 키워드 3가지만 출력할 것.
- {[review_line, review_line, review_line]}는 당신이 투표들과 댓글들을 상세하게 분석한 총평을 최소 50자 이상으로 3가지 출력할 것.
######################################################

[OUTPUT]
"""

        response = self._generate_analysis(prompt, model, tokenizer, device)
        result = parse_moderation_response(response)
        return result

    def _summarize_group_data(self, df):
        group_id = df["group_id"].iloc[0]
        group_name = df["group_name"].iloc[0]
        vote_summary = ", ".join([str(v) for v in df["vote"].unique() if str(v).strip() not in ["None", "nan", ""]])
        comment_summary = ", ".join([str(c) for c in df["comment"].unique() if c and c != "None"])
        if not comment_summary:
            comment_summary = "없음"
        return {
            "group_id": group_id,
            "group_name": group_name,
            "vote_summary": vote_summary,
            "comment_summary": comment_summary
        }

    def generate(self, start_date: str, end_date: str, model, tokenizer, device, logger) -> dict:
        logger.info(f"Analysis 시작: start_date[{start_date}] ~ end_date[{end_date}]")

        # 그룹 정보 조회
        group_data_df = self._get_group_data(start_date, end_date, logger)
        if group_data_df is None:
            return {
                    "analysis_results": None
            }

        group_analysis = []

        for group_id, group_df in group_data_df.groupby("group_id"):
            logger.info(f"그룹[{group_id}] 투표 및 댓글 데이터 분석 시작")

            summary_str = self._summarize_group_data(group_df)
            analysis = self._analyze_group(summary_str, model, tokenizer, device)
            
            analysis = analysis.replace('```json', '')
            analysis = analysis.replace('```', '')
            analysis = analysis.strip()

            group_analysis.append(analysis)

            try:
                parsed_input = json.loads(analysis)
            except Exception as e:
                logger.error(f"[ERROR] JSON 파싱 실패: {e}, 원본: {analysis}")
                continue

            # key 이름 변환
            parsed_input = {
                "groupId": parsed_input["group_id"],
                "groupName": parsed_input["group_name"],
                "voteSummary": parsed_input["vote_summary"],
                "commentSummary": parsed_input["comment_summary"],
                "emotion": parsed_input["emotion"],
                "topKeywords": parsed_input["top_keywords"],
                "modelReview": parsed_input["modelReview"]
            }

            # request body 생성
            request_body = {
                "groupId": parsed_input["groupId"],
                "weekStartAt": start_date,
                "overview": {
                    "voteSummary": parsed_input["voteSummary"],
                    "commentSummary": parsed_input["commentSummary"]
                },
                "sentiment": {
                    "emotion": parsed_input["emotion"],
                    "topKeywords": parsed_input["topKeywords"]
                },
                "modelReview": parsed_input["modelReview"],
                "version": MODEL_VERSION
            }

            logger.info(f"BE 투표 분석 결과 등록 요청 시작")

            url = f"http://{BE_SERVER_IP}:{BE_SERVER_PORT}/api/v1/ai/groups/analysis"
            headers = {"Content-Type": "application/json"}

            try:
                response = requests.post(url, headers=headers, data=json.dumps(request_body))
                code = response.status_code

                logger.info(f"Status code: {response.status_code}")
                logger.info(f"Response body: {response.text}")

                if code == 201:
                    logger.info("201 Created: SUCCESS - 분석 데이터 등록 완료")
                elif code == 400:
                    body = response.json()
                    message = body.get("message", "UNKNOWN")
                    if message == "INVALID_TIME":
                        logger.warning("400 Bad Request: INVALID_TIME - weekStartAt 유효성 오류")
                    elif message == "INVALID_ERROR":
                        logger.warning("400 Bad Request: INVALID_ERROR - 필수값 누락 또는 형식 오류")
                    else:
                        logger.warning(f"400 Bad Request: 기타 오류 ({message})")
                elif code == 404:
                    logger.error("404 Not Found: GROUP_NOT_FOUND - 해당 그룹 없음")
                elif code == 409:
                    logger.warning("409 Conflict: DUPLICATE_ANALYSIS - 중복 데이터")
                elif code == 500:
                    body = response.json()
                    message = body.get("message", "UNKNOWN")
                    if message == "MONGO_SAVE_FAILED":
                        logger.error("500 Internal Server Error: MONGO_SAVE_FAILED - DB 저장 오류")
                    else:
                        logger.error(f"500 Internal Server Error: {message} - 서버 내부 오류")
                else:
                    logger.error(f"{code} Unexpected response: {response.text}")
            
            except Exception as e:
                logger.exception(f"API 요청 중 예외 발생: {e}")

        logger.info(f"Analysis 완료.")

        return {
            "analysis_results": group_analysis
        }
