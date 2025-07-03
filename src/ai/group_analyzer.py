import json
import os
from ai.moderation_llm import parse_moderation_response
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
            AND v.created_at BETWEEN '{start_date} 00:00:00' AND '{end_date} 00:00:00'
        GROUP BY
            g.id,
            g.name,
            v.id"""

        # pandas로 SQL 실행 결과 바로 읽기
        group_data_df = pd.read_sql(query, con=engine)
        group_data_df['yes'] = group_data_df['yes'].fillna(0)
        group_data_df['no'] = group_data_df['no'].fillna(0)

        logger.info("데이터 조회 결과")
        logger.info(group_data_df)

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
You are a voting‑data analyst.
Return ONE plain‑text report only.
Output MUST be placed strictly between [ANSWER] and [/ANSWER].
Do NOT repeat or mention the prompt, template, or any instructions.

**DATA는 Pandas의 DataFrame을 문자열로 변환한 것이고, Comment는 Comma Separated Field임을 참고할 것.**
**개요와 총평은 반드시 서술형 구어체 만을 사용하여 작성할 것.(example. ~~입니다. ~~할 것입니다. 등)**
######################################################

#######################  USER  #######################
[DATA]
""" + f"{group}" + """
[/DATA]

Fill every {placeholder} in the template below with real analysis results.
If information is missing, write '없음' (no quotes).
Write **at least 3 lines** in section 3.
Keep all other words and line‑breaks exactly as‑is.

[FORMAT]
[ANSWER]
분석 보고서

1. 개요
- 그룹명: {group_name}
- 투표 내용 요약: {vote_summary}
- 댓글 내용 요약: {comment_summary}

2. 그룹 전체 분위기
- 감성: {sentiment_overall}
- 주요 키워드: {kw1}, {kw2}, {kw3}

3. 총평
- {review_line1}
- {review_line2}
- {review_line3}
[/ANSWER]
######################################################
"""

        response = self._generate_analysis(prompt, model, tokenizer, device)
        result = parse_moderation_response(response)
        return result

    def generate(self, start_date: str, end_date: str, model, tokenizer, device, logger) -> dict:
        logger.info(f"Analysis 시작: start_date[{start_date}] ~ end_date[{end_date}]")

        # 그룹 정보 조회
        group_data_df = self._get_group_data(start_date, end_date)
        groups = [data.copy() for _, data in group_data_df.groupby('group_id')]
        
        # 그룹별 분석

        logger.info(f"그룹별 투표 및 댓글 데이터 분석 시작")
        group_analysis = []
        for group in groups:
            analysis = self._analyze_group(group, model, tokenizer, device)
            group_analysis.append(analysis)

            logger.info(f"분석 완료, 결과 후처리 진행 중...")
            analysis = analysis.replace('[ANSWER]', '')
            analysis = analysis.replace('[/ANSWER]', '')
            analysis = analysis.replace('```', '')
            analysis = analysis.replace('# End of File', '')
            analysis = analysis.strip()

            print(analysis)
            print()
            logger.info(f"Group Analysis 완료: {analysis}")

        logger.info(f"그룹별 투표 및 댓글 데이터 분석 완료")

        # 그룹별 분석 결과 데이터 구조화 -> to JSON
        logger.info(f"분석 데이터 후처리(to JSON) 및 BE 전송 시작")
        for input_str in group_analysis:
            logger.info(f"분석 데이터 후처리 작업 중...")
            section_pattern = re.compile(r"(\d+)\.\s*([^\n]+)\n((?:\s+-[^\n]*\n?)+)", re.MULTILINE)
            matches = section_pattern.findall(input_str)

            result = {}
            for num, section, body in matches:
                section_name = section.strip()
                
                if section_name == "총평":
                    items = [
                        re.sub(r'^\s*-\s*', '', line).strip()
                        for line in body.strip().split('\n') if line.strip()
                    ]
                    result[section_name] = items
                else:
                    section_dict = {}
                    for line in body.strip().split('\n'):
                        line = line.strip()
                        if not line.startswith('-'):
                            continue
                        content = line[1:].strip()
                        if ':' in content:
                            title, value = content.split(':', 1)
                            section_dict[title.strip()] = value.strip()
                    result[section_name] = section_dict

            logger.info(json.dumps(result, ensure_ascii=False, indent=2))
        
            # TODO: BE에 분석 리포트 등록 요청 보내기
            # logger.info(f"분석 데이터 BE 전송 중...")
            # NOTE: 전송 Delay가 필요할 수도 있음. -> 분석 루프에서 바로 후처리 하고 송신하는 방법도 고려.
        
        return {
            "analysis_results": group_analysis
        }
