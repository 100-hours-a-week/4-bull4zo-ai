import logging
import os
import csv
import hashlib
import datetime
import queue
import stat
from logging.handlers import TimedRotatingFileHandler, QueueHandler, QueueListener
from multiprocessing import Queue
from pathlib import Path
from typing import Dict, Any, Optional, Union


class CSVFormatter(logging.Formatter):
    """CSV 형식으로 로그를 포맷팅하는 클래스"""
    
    def __init__(self):
        super().__init__()
        self.fields = ['date', 'time', 'level', 'section', 'request_id', 
                      'user_id', 'pred_label', 'pred_score', 'model_version', 'message']
        
    def format(self, record):
        # 기본 필드 설정
        date_time = datetime.datetime.fromtimestamp(record.created)
        date_str = date_time.strftime('%Y-%m-%d')
        time_str = date_time.strftime('%H:%M:%S.%f')[:-3]  # 밀리초까지
        
        # extra 필드 기본값 설정
        section = getattr(record, 'section', '-')
        request_id = getattr(record, 'request_id', '-')
        user_id = getattr(record, 'user_id', '-')
        pred_label = getattr(record, 'pred_label', '-')
        pred_score = getattr(record, 'pred_score', '-')
        model_version = getattr(record, 'model_version', '-')
        
        # 메시지
        message = record.getMessage()
        
        # content 필드가 있고 moderation 섹션인 경우 해시 처리
        if section == 'moderation' and hasattr(record, 'content'):
            content = getattr(record, 'content', '')
            if content and isinstance(content, str):
                # 해시 생성 (16자리)
                content_hash = hashlib.md5(content.encode('utf-8')).hexdigest()[:16]
                setattr(record, 'content_hash', content_hash)
        
        # CSV 행 생성
        csv_row = [
            date_str, time_str, record.levelname, section, request_id,
            user_id, pred_label, pred_score, model_version, message
        ]
        
        # CSV 문자열로 변환
        return ','.join(str(field) for field in csv_row)


class CSVFileHandler(TimedRotatingFileHandler):
    """CSV 파일에 로그를 기록하는 핸들러"""
    
    def __init__(self, filename, when='midnight', backupCount=14):
        super().__init__(
            filename=filename,
            when=when,
            backupCount=backupCount,
            encoding='utf-8'
        )
        self.header_written = False
        self.formatter = CSVFormatter()
        
        # 파일 권한 설정 (600)
        try:
            os.chmod(filename, stat.S_IRUSR | stat.S_IWUSR)
        except FileNotFoundError:
            # 파일이 아직 없는 경우
            pass
    
    def emit(self, record):
        if self.stream is None:
            self.stream = self._open()
            
            # 새 파일에는 헤더 추가
            if not self.header_written:
                writer = csv.writer(self.stream)
                writer.writerow(self.formatter.fields)
                self.stream.flush()
                self.header_written = True
                
                # 파일 생성 후 권한 설정
                os.chmod(self.baseFilename, stat.S_IRUSR | stat.S_IWUSR)
        
        super().emit(record)
    
    def doRollover(self):
        super().doRollover()
        self.header_written = False


class SecureFileHandler(logging.FileHandler):
    """보안 설정이 적용된 파일 핸들러"""
    
    def __init__(self, filename, mode='a', encoding='utf-8'):
        super().__init__(filename, mode, encoding)
        
        # 파일 권한 설정 (600)
        try:
            os.chmod(filename, stat.S_IRUSR | stat.S_IWUSR)
        except FileNotFoundError:
            # 파일이 아직 없는 경우
            pass
    
    def _open(self):
        stream = super()._open()
        
        # 파일 생성 후 권한 설정
        os.chmod(self.baseFilename, stat.S_IRUSR | stat.S_IWUSR)
        return stream


class EmailHandler(logging.Handler):
    """이메일로 오류 알림을 보내는 핸들러"""
    
    def __init__(self, level=logging.ERROR):
        super().__init__(level)
    
    def emit(self, record):
        # 이메일 발송 로직은 환경에 따라 구현
        # SMTP 서버, 수신자 등은 환경 변수로 설정 가능
        pass


class SafeExtraFormatter(logging.Formatter):
    """필수 extra 필드가 없을 때 기본값을 채워주는 포맷터"""
    def format(self, record):
        # 누락된 필드에 기본값 할당
        for field, default in [
            ("section", "-"),
            ("request_id", "-"),
            ("content_hash", "-"),
            ("user_id", "-"),
            ("pred_label", "-"),
            ("pred_score", "-"),
            ("model_version", "-")
        ]:
            if not hasattr(record, field):
                setattr(record, field, default)
        return super().format(record)


# 글로벌 큐 리스너 참조
_queue_listeners = {}


def init_process_logging(proc: str = "ai") -> logging.Logger:
    """
    프로세스별 로깅 시스템을 초기화합니다.
    
    Args:
        proc: 프로세스 이름 ("ai" 또는 "api")
        
    Returns:
        logging.Logger: 기본 로거
    """
    if proc not in ["ai", "api"]:
        raise ValueError("proc must be 'ai' or 'api'")
    
    # 로그 디렉터리 설정
    log_dir = Path(f"logs/{proc}")
    os.makedirs(log_dir, exist_ok=True)
    
    # 로거 설정
    logger = logging.getLogger(proc)
    
    # 이미 설정된 로거인 경우 반환
    if logger.handlers:
        return logger
    
    # 로그 레벨 설정
    logger.setLevel(logging.INFO)
    logger.propagate = False  # 부모 로거로 전파 방지
    
    # 멀티프로세스 큐 설정
    log_queue = Queue()
    
    # CSV 핸들러 설정 (날짜별 자동 교체)
    csv_filename = log_dir / f"{datetime.datetime.now().strftime('%Y-%m-%d')}.csv"
    csv_handler = CSVFileHandler(
        filename=csv_filename,
        when="midnight",
        backupCount=14
    )
    csv_handler.setLevel(logging.INFO)
    
    # moderation.log 핸들러 설정
    moderation_handler = SecureFileHandler(
        filename=log_dir / "moderation.log",
        encoding="utf-8"
    )
    moderation_handler.setLevel(logging.INFO)
    moderation_handler.setFormatter(SafeExtraFormatter(
        '%(asctime)s,%(levelname)s,%(section)s,%(request_id)s,%(content_hash)s,%(message)s'
    ))
    
    # 모더레이션 필터 - section이 'moderation'인 로그만 허용
    class ModerationFilter(logging.Filter):
        def filter(self, record):
            return getattr(record, 'section', '') == 'moderation'
    
    moderation_handler.addFilter(ModerationFilter())
    
    # error.log 핸들러 설정
    error_handler = SecureFileHandler(
        filename=log_dir / "error.log",
        encoding="utf-8"
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(SafeExtraFormatter(
        '%(asctime)s [%(levelname)s] %(section)s:%(request_id)s - %(message)s'
    ))
    
    # 이메일 알림 핸들러 (오류 발생 시)
    email_handler = EmailHandler(level=logging.ERROR)
    email_handler.setFormatter(SafeExtraFormatter(
        '[%(levelname)s] %(section)s:%(request_id)s - %(message)s'
    ))
    
    # 콘솔 핸들러 (개발/디버깅용)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(SafeExtraFormatter(
        '%(asctime)s [%(levelname)s] %(section)s:%(request_id)s - %(message)s'
    ))
    
    # QueueListener 설정 (모든 핸들러에 대해)
    handlers = [csv_handler, moderation_handler, error_handler, console_handler, email_handler]
    queue_listener = QueueListener(log_queue, *handlers, respect_handler_level=True)
    queue_listener.start()
    
    # 글로벌 참조에 저장
    _queue_listeners[proc] = queue_listener
    
    # QueueHandler 설정
    queue_handler = QueueHandler(log_queue)
    logger.addHandler(queue_handler)
    
    return logger


def shutdown_logging(proc: str = "ai"):
    """
    프로세스별 로깅 시스템을 종료합니다.
    
    Args:
        proc: 프로세스 이름 ("ai" 또는 "api")
    """
    if proc in _queue_listeners:
        _queue_listeners[proc].stop()
        del _queue_listeners[proc]


# 로그 메시지 헬퍼 함수
def get_content_hash(content: str) -> str:
    """
    콘텐츠의 해시값을 생성합니다(16자리).
    
    Args:
        content: 해시할 텍스트 콘텐츠
        
    Returns:
        str: 16자리 MD5 해시
    """
    return hashlib.md5(content.encode('utf-8')).hexdigest()[:16] 
