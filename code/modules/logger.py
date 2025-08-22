# -*- coding: utf-8 -*-
"""
로그 유틸리티. 실행중인 "스크립트명.log" 파일에 로그 출력.
사용법:
from modules.logger import LoggerManager

log = LoggerManager("ModuleName")
log.info("로그 메시지 출력")
log.log_function_start("function_name", param1="value")

모든 LoggerManager 인스턴스는 동일한 전역 로거를 공유합니다.
"""

import os
import logging
from datetime import datetime, timezone, timedelta
from typing import Optional


class CustomFormatter(logging.Formatter):
    """한국 시간대(UTC+9)로 로그 포맷을 설정하는 커스텀 포매터"""
    def format(self, record):
        # UTC+9 서울 타임존 설정
        kst = timezone(timedelta(hours=9))
        timestamp = datetime.fromtimestamp(record.created, tz=kst).strftime("%y-%m-%d %H:%M:%S")
        return f"[{timestamp}] [INFO] {record.getMessage()}"


def setup_logger(script_file_path=None, file_mode="w"):
    """
    로거를 설정하고 반환합니다.
    
    Args:
        script_file_path (str, optional): 스크립트 파일 경로. None이면 현재 실행중인 스크립트 경로 사용
        file_mode (str): 파일 핸들러 모드 ("w", "a" 등)
    
    Returns:
        logging.Logger: 설정된 로거 객체
    """
    # 로거 설정
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    
    # 기존 핸들러가 있다면 제거 (중복 방지)
    if logger.handlers:
        logger.handlers.clear()
    
    # 콘솔 핸들러
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(CustomFormatter())
    
    # 파일 핸들러 (스크립트명.log로 저장)
    if script_file_path is None:
        import sys
        # __main__ 모듈의 __file__ 속성을 사용하여 실제 실행중인 스크립트 경로 얻기
        import __main__
        if hasattr(__main__, '__file__'):
            script_file_path = __main__.__file__
        else:
            script_file_path = sys.argv[0] if sys.argv[0] else __file__
    
    # 절대 경로로 변환
    script_file_path = os.path.abspath(script_file_path)
    script_name = os.path.splitext(os.path.basename(script_file_path))[0]
    script_dir = os.path.dirname(script_file_path)
    
    # 로그 파일이 logger.py의 이름으로 저장되는 것을 방지
    if script_name == 'logger':
        import __main__
        if hasattr(__main__, '__file__'):
            main_script_path = os.path.abspath(__main__.__file__)
            script_name = os.path.splitext(os.path.basename(main_script_path))[0]
            script_dir = os.path.dirname(main_script_path)
    
    log_file_path = os.path.join(script_dir, f"{script_name}.log")
    
    # 로그 파일의 디렉토리가 없으면 생성
    log_dir = os.path.dirname(log_file_path)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    
    file_handler = logging.FileHandler(log_file_path, mode=file_mode, encoding='utf-8')
    file_handler.setFormatter(CustomFormatter())
    
    # 핸들러 추가
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    logger.propagate = False
    
    return logger


# 전역 로거 초기화 (한 번만 실행)
_logger = None
_logger_initialized = False

def get_global_logger():
    """전역 로거를 반환합니다. 초기화되지 않았다면 초기화합니다."""
    global _logger, _logger_initialized
    
    if not _logger_initialized:
        _logger = setup_logger()
        _logger_initialized = True
    
    return _logger 


class LoggerManager:
    """로깅 관리 클래스"""
    
    def __init__(self, module_name: str = None):
        """
        LoggerManager 초기화
        
        Args:
            module_name (str, optional): 모듈 이름 (로그에 표시될 카테고리명)
        """
        self.module_name = module_name or "module"
        self.logger = get_global_logger()
    
    def info(self, *args):
        """정보 레벨 로그"""
        message = f"[{self.module_name}] " + ' '.join(str(arg) for arg in args)
        self.logger.info(message)
    
    def debug(self, *args):
        """디버그 레벨 로그"""
        message = f"[{self.module_name}] " + ' '.join(str(arg) for arg in args)
        self.logger.debug(message)
    
    def warning(self, *args):
        """경고 레벨 로그"""
        message = f"[{self.module_name}] " + ' '.join(str(arg) for arg in args)
        self.logger.warning(message)
    
    def error(self, *args):
        """오류 레벨 로그"""
        message = f"[{self.module_name}] " + ' '.join(str(arg) for arg in args)
        self.logger.error(message)
    
    def critical(self, *args):
        """심각한 오류 레벨 로그"""
        message = f"[{self.module_name}] " + ' '.join(str(arg) for arg in args)
        self.logger.critical(message)
    
    def log_function_start(self, function_name: str, **kwargs):
        """함수 시작 로그"""
        params = ', '.join([f"{k}={v}" for k, v in kwargs.items()])
        self.info(f"📍 {function_name} 시작" + (f" - 매개변수: {params}" if params else ""))
    
    def log_function_end(self, function_name: str, result=None):
        """함수 종료 로그"""
        if result is not None:
            self.info(f"✅ {function_name} 완료 - 결과: {result}")
        else:
            self.info(f"✅ {function_name} 완료")
    
    def log_error(self, function_name: str, error: Exception):
        """에러 로그"""
        self.error(f"❌ {function_name} 오류: {str(error)}")
    
    def log_step(self, step_name: str, details: str = None):
        """단계별 진행 로그"""
        if details:
            self.info(f"🔄 {step_name}: {details}")
        else:
            self.info(f"🔄 {step_name}")
    
    def log_success(self, message: str):
        """성공 로그"""
        self.info(f"✅ {message}")
    
    def log_warning_with_icon(self, message: str):
        """경고 로그 (아이콘 포함)"""
        self.warning(f"⚠️ {message}")
    
    def log_error_with_icon(self, message: str):
        """오류 로그 (아이콘 포함)"""
        self.error(f"❌ {message}")
    
    @staticmethod
    def get_global_logger():
        """전역 로거 반환 (기존 호환성)"""
        return get_global_logger()