"""
로깅 관리 모듈

이 모듈은 기존 utils/log_util.py의 기능을 클래스 기반으로 래핑하여
모듈화된 구조에서 사용할 수 있도록 합니다.

주요 기능:
1. 기존 log_util.py 기능 래핑
2. 클래스 기반 로깅 인터페이스 제공
3. 모듈별 로그 파일 관리
"""

import os
import sys
from pathlib import Path
from typing import Optional

# 상위 디렉토리의 utils 모듈을 import하기 위한 경로 추가
sys.path.append(str(Path(__file__).parent.parent))

try:
    import utils.log_util as log
except ImportError:
    # utils 모듈을 찾을 수 없는 경우 기본 로깅 사용
    import logging
    
    class BasicLogger:
        def __init__(self):
            self.logger = logging.getLogger(__name__)
            self.logger.setLevel(logging.INFO)
            handler = logging.StreamHandler()
            formatter = logging.Formatter('[%(asctime)s] [INFO] %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        
        def info(self, *args):
            message = ' '.join(str(arg) for arg in args)
            self.logger.info(message)
        
        def debug(self, *args):
            message = ' '.join(str(arg) for arg in args)
            self.logger.debug(message)
        
        def warning(self, *args):
            message = ' '.join(str(arg) for arg in args)
            self.logger.warning(message)
        
        def error(self, *args):
            message = ' '.join(str(arg) for arg in args)
            self.logger.error(message)
        
        def critical(self, *args):
            message = ' '.join(str(arg) for arg in args)
            self.logger.critical(message)
    
    log = BasicLogger()


class LoggerManager:
    """로깅 관리 클래스"""
    
    def __init__(self, module_name: str = None, file_mode: str = "w"):
        """
        LoggerManager 초기화
        
        Args:
            module_name (str, optional): 모듈 이름 (로그 파일명에 사용)
            file_mode (str): 파일 모드 ("w": 덮어쓰기, "a": 추가)
        """
        self.module_name = module_name or "module"
        self.file_mode = file_mode
        
        # 기존 log_util 설정 변경 (필요시)
        if hasattr(log, 'change_file_mode'):
            log.change_file_mode(file_mode)
    
    def info(self, *args):
        """정보 레벨 로그"""
        if self.module_name:
            message = f"[{self.module_name}] " + ' '.join(str(arg) for arg in args)
            log.info(message)
        else:
            log.info(*args)
    
    def debug(self, *args):
        """디버그 레벨 로그"""
        if self.module_name:
            message = f"[{self.module_name}] " + ' '.join(str(arg) for arg in args)
            log.debug(message)
        else:
            log.debug(*args)
    
    def warning(self, *args):
        """경고 레벨 로그"""
        if self.module_name:
            message = f"[{self.module_name}] " + ' '.join(str(arg) for arg in args)
            log.warning(message)
        else:
            log.warning(*args)
    
    def error(self, *args):
        """오류 레벨 로그"""
        if self.module_name:
            message = f"[{self.module_name}] " + ' '.join(str(arg) for arg in args)
            log.error(message)
        else:
            log.error(*args)
    
    def critical(self, *args):
        """심각한 오류 레벨 로그"""
        if self.module_name:
            message = f"[{self.module_name}] " + ' '.join(str(arg) for arg in args)
            log.critical(message)
        else:
            log.critical(*args)
    
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
        """전역 로거 반환 (기존 log_util 호환성)"""
        return log