import time
import json
import requests
from src.api.dtos.moderation_result_request import ModerationResultRequest
from src.ai.moderation_utils import get_relevant_context, validate_spec, CATEGORY_MAPPING, CATEGORY_ALIASES, normalize_category

def build_moderation_prompt(moderation_request, relevant_context):
    content = moderation_request.content
    chat = []
    chat.append({
        "role": "system",
        "content": (
            "[시스템 지침] 반드시 아래 규칙을 따르세요.\n"
            "- 반드시 '[카테고리]: [사유]' 형식으로만 답변하세요. (예: 욕설/비방: 욕설이 포함되어 있어 부적절합니다.)\n"
            "- 부적절한 내용일 경우 '[카테고리]'는 반드시 아래 값 중 하나여야 합니다:\n"
            "  욕설/비방, 정치, 음란성/선정성, 스팸/광고, 사칭/사기/개인정보 노출, 기타\n"
            "- '[사유]'는 해당 카테고리로 분류한 이유를 구체적으로 작성하세요.\n"
            "- 적절한 내용일 경우 반드시 '검열 불필요: 적절한 표현입니다.'로만 답변하세요. (예: 검열 불필요: 적절한 표현입니다.)\n"
            "- 절대 '검열 필요:', '부적절', '적절', '카테고리:' 등 다른 형식이나 영어로 시작하지 마세요.\n"
            "- 반드시 위에 제시된 한글 카테고리(정확히 일치)로만 시작하세요.\n"
            "\n"
            "아래 계층별 지침을 반드시 우선순위대로 적용하세요.\n"
            "==== 시스템 지침 끝 ===="
        )
    })
    if relevant_context:
        chat.append({
            "role": "system",
            "content": (
                "[RAG 컨텍스트] 아래는 참고용 실제 사례입니다. 반드시 참고만 하세요.\n"
                f"{relevant_context}\n==== RAG 컨텍스트 끝 ===="
            )
        })
    chat.append({
        "role": "user",
        "content": f"{content}"
    })
    return chat

def run_llm_inference(chat, model, tokenizer, device):
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
    result = tokenizer.batch_decode(output_ids)[0]
    return result, inference_time

def parse_moderation_response(raw_response):
    try:
        start_marker = "<|im_start|>assistant\n"
        end_marker = "<|im_end|>"
        start_positions = []
        pos = 0
        while True:
            pos = raw_response.find(start_marker, pos)
            if pos == -1:
                break
            start_positions.append(pos)
            pos += 1
        if start_positions:
            last_start = start_positions[-1] + len(start_marker)
            last_end = raw_response.find(end_marker, last_start)
            if last_end != -1:
                result = raw_response[last_start:last_end].strip()
            else:
                result = raw_response[last_start:].strip()
        else:
            result = raw_response.strip()
    except Exception:
        result = raw_response.strip()
    return result

def extract_category_and_reason(result):
    kr_categories = list(CATEGORY_MAPPING.keys())
    found_category_kr = "기타"
    reason_detail = ""
    if ": " in result:
        category_part, reason_detail = result.split(": ", 1)
        if any(keyword in reason_detail for keyword in ["성적으로 암시", "성적", "음란", "선정"]):
            found_category_kr = "음란성/선정성"
        else:
            category_part_norm = category_part.replace(" ", "")
            if category_part_norm in CATEGORY_ALIASES:
                found_category_kr = CATEGORY_ALIASES[category_part_norm]
            else:
                for kr in kr_categories:
                    if category_part_norm in kr.replace(" ", ""):
                        found_category_kr = kr
                        break
        if not reason_detail or reason_detail.strip() in kr_categories:
            reason_detail = "부적절한 내용이 감지되었습니다."
    else:
        if result.startswith("검열 필요:"):
            category_kr = result.replace("검열 필요:", "").strip()
            found_category_kr = category_kr if category_kr in kr_categories else "기타"
            reason_detail = "부적절한 내용이 감지되었습니다."
        else:
            result_no_space = result.replace(" ", "")
            for category in kr_categories:
                if category.replace(" ", "") in result_no_space:
                    found_category_kr = category
                    break
            temp_result = result
            for category in kr_categories:
                temp_result = temp_result.replace(category, "", 1)
            reason_detail = temp_result.strip()
            if not reason_detail or reason_detail in kr_categories:
                reason_detail = "부적절한 내용이 감지되었습니다."
    norm_found_category_kr = normalize_category(found_category_kr)
    if norm_found_category_kr in CATEGORY_ALIASES:
        found_category_kr = CATEGORY_ALIASES[norm_found_category_kr]
    found_category_en = CATEGORY_MAPPING.get(found_category_kr, "OTHER")
    return found_category_kr, found_category_en, reason_detail

def send_moderation_callback(moderation_result_request, callback_url, logger, request_id):
    headers = {"Content-Type": "application/json"}
    try:
        logger.info(f"검열 결과 전송 시작", extra={"section": "server", "request_id": request_id})
        response = requests.post(callback_url, json=moderation_result_request.dict(), headers=headers)
        if response.status_code == 201:
            logger.info(f"검열 결과 전송 성공: HTTP {response.status_code}", extra={"section": "server", "request_id": request_id})
        else:
            logger.error(f"검열 결과 전송 실패: HTTP {response.status_code}", extra={"section": "server", "request_id": request_id})
    except requests.exceptions.RequestException as e:
        error_msg = f"검열 결과 전송 중 네트워크 오류: {str(e)}"
        logger.error(error_msg, exc_info=True, extra={"section": "server", "request_id": request_id})

def moderation_pipeline(moderation_request, model, tokenizer, device, callback_url, logger):
    content = moderation_request.content
    request_id = str(moderation_request.voteId)
    try:
        logger.info(f"검열 요청 처리 시작", extra={"section": "moderation", "request_id": request_id, "content": content})
        relevant_context = get_relevant_context(content)
        chat = build_moderation_prompt(moderation_request, relevant_context)
        raw_response, inference_time = run_llm_inference(chat, model, tokenizer, device)
        result = parse_moderation_response(raw_response)
        if not validate_spec(result):
            logger.warning(f"모델 응답이 스펙을 벗어남: '{result}'", extra={"section": "moderation", "request_id": request_id})
            result = "기타: 출력 스펙을 위반한 응답입니다."
        if not result:
            logger.warning("모델 응답이 비어있습니다.", extra={"section": "moderation", "request_id": request_id})
        logger.info(json.dumps({
            "vote_id": moderation_request.voteId,
            "content": content,
            "model_response": result,
            "inference_time": f"{inference_time:.2f}s"
        }, ensure_ascii=False), extra={"section": "moderation", "request_id": request_id, "model_version": "v1.0.0"})
        if result.strip().startswith("검열 불필요: 적절한 표현입니다"):
            final_result = "APPROVED"
            final_reason = "NONE"
            final_reason_detail = "적절한 표현입니다"
            pred_label = "APPROVED"
            pred_score = "1.0"
        else:
            final_result = "REJECTED"
            found_category_kr, found_category_en, reason_detail = extract_category_and_reason(result)
            final_reason = found_category_en
            final_reason_detail = reason_detail
            pred_label = found_category_en
            pred_score = "0.9"
        version = "1.0.0"
        moderation_result_request = ModerationResultRequest(
            voteId=moderation_request.voteId if moderation_request.voteId is not None else 0,
            result=final_result,
            reason=final_reason,
            reasonDetail=final_reason_detail,
            version=version
        )
        logger.info(f"검열 결과: {final_result}, 카테고리={final_reason}, 이유='{final_reason_detail}'", extra={"section": "server", "request_id": request_id, "pred_label": pred_label, "pred_score": pred_score, "model_version": "v1.0.0"})
        send_moderation_callback(moderation_result_request, callback_url, logger, request_id)
        return moderation_result_request.dict()
    except Exception as e:
        error_msg = f"검열 처리 중 오류 발생: {str(e)}"
        logger.error(error_msg, exc_info=True, extra={"section": "moderation", "request_id": request_id})
        return {"result": "ERROR"} 
