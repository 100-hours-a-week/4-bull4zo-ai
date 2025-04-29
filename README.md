# MOA AI Server

## Summary

이 Repository는 MOA 프로젝트의 AI Server 프로젝트입니다.

LLM을 기반으로 투표 내용을 검열하거나, 흥미로운 투표 생성, 투표/댓글 분석 리포트 생성을 담당합니다.

### References

- [MOA Wiki HOME](https://github.com/100-hours-a-week/4-bull4zo-wiki/wiki)
- [MOA AI Wiki](https://github.com/100-hours-a-week/4-bull4zo-wiki/wiki/AI-Wiki)

## Development Environment

- Programming Language : Python v3.9.22

## LLM (Large Language Model)

- HyperCLOVA X SEED 3B

## Project Directory Structure
```
4-bull4zo-ai
- app.py
- src
    - ai
        - ai_process.py
    - api
        - api_process.py
        - controllers
            - moderation_controller.py
            - status_controller.py
        - dtos
            - moderation_request.py
```

## How to run
run command in terminal followings:
```
> python app.py
```

### running process
1. start app.py
2. run `ai_process.py` and `api_process.py` by multiprocessing
3. `ai_process.py` runs model loading -> processing queue
4. `api_process.py` run FastAPI Server
