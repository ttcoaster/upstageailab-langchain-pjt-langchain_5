"""
LoggerManager 테스트

로깅 관리 기능을 테스트합니다.
"""

import pytest
import tempfile
import os
from pathlib import Path
import sys
from unittest.mock import patch, MagicMock

# 현재 파일의 부모 디렉토리를 sys.path에 추가
current_dir = Path(__file__).parent.parent
sys.path.insert(0, str(current_dir))

from modules.logger import LoggerManager


class TestLoggerManager:
    """LoggerManager 테스트 클래스"""
    
    @pytest.fixture
    def logger_manager(self):
        """LoggerManager 인스턴스 픽스처"""
        return LoggerManager(module_name="TestModule")
    
    @patch('modules.logger.log')
    def test_info_logging(self, mock_log, logger_manager):
        """정보 로그 테스트"""
        logger_manager.info("테스트", "메시지")
        mock_log.info.assert_called_once_with("[TestModule] 테스트 메시지")
    
    @patch('modules.logger.log')
    def test_debug_logging(self, mock_log, logger_manager):
        """디버그 로그 테스트"""
        logger_manager.debug("디버그", "메시지")
        mock_log.debug.assert_called_once_with("[TestModule] 디버그 메시지")
    
    @patch('modules.logger.log')
    def test_warning_logging(self, mock_log, logger_manager):
        """경고 로그 테스트"""
        logger_manager.warning("경고", "메시지")
        mock_log.warning.assert_called_once_with("[TestModule] 경고 메시지")
    
    @patch('modules.logger.log')
    def test_error_logging(self, mock_log, logger_manager):
        """오류 로그 테스트"""
        logger_manager.error("오류", "메시지")
        mock_log.error.assert_called_once_with("[TestModule] 오류 메시지")
    
    @patch('modules.logger.log')
    def test_critical_logging(self, mock_log, logger_manager):
        """심각한 오류 로그 테스트"""
        logger_manager.critical("심각한", "오류")
        mock_log.critical.assert_called_once_with("[TestModule] 심각한 오류")
    
    @patch('modules.logger.log')
    def test_log_function_start(self, mock_log, logger_manager):
        """함수 시작 로그 테스트"""
        logger_manager.log_function_start("test_function", param1="value1", param2="value2")
        expected_message = "[TestModule] 📍 test_function 시작 - 매개변수: param1=value1, param2=value2"
        mock_log.info.assert_called_once_with(expected_message)
    
    @patch('modules.logger.log')
    def test_log_function_end(self, mock_log, logger_manager):
        """함수 종료 로그 테스트"""
        logger_manager.log_function_end("test_function", result="success")
        expected_message = "[TestModule] ✅ test_function 완료 - 결과: success"
        mock_log.info.assert_called_once_with(expected_message)
    
    @patch('modules.logger.log')
    def test_log_function_end_no_result(self, mock_log, logger_manager):
        """결과 없는 함수 종료 로그 테스트"""
        logger_manager.log_function_end("test_function")
        expected_message = "[TestModule] ✅ test_function 완료"
        mock_log.info.assert_called_once_with(expected_message)
    
    @patch('modules.logger.log')
    def test_log_error(self, mock_log, logger_manager):
        """에러 로그 테스트"""
        test_error = Exception("테스트 에러")
        logger_manager.log_error("test_function", test_error)
        expected_message = "[TestModule] ❌ test_function 오류: 테스트 에러"
        mock_log.error.assert_called_once_with(expected_message)
    
    @patch('modules.logger.log')
    def test_log_step(self, mock_log, logger_manager):
        """단계 로그 테스트"""
        logger_manager.log_step("초기화", "설정 로드 중")
        expected_message = "[TestModule] 🔄 초기화: 설정 로드 중"
        mock_log.info.assert_called_once_with(expected_message)
    
    @patch('modules.logger.log')
    def test_log_step_no_details(self, mock_log, logger_manager):
        """세부사항 없는 단계 로그 테스트"""
        logger_manager.log_step("초기화")
        expected_message = "[TestModule] 🔄 초기화"
        mock_log.info.assert_called_once_with(expected_message)
    
    @patch('modules.logger.log')
    def test_log_success(self, mock_log, logger_manager):
        """성공 로그 테스트"""
        logger_manager.log_success("초기화 완료")
        expected_message = "[TestModule] ✅ 초기화 완료"
        mock_log.info.assert_called_once_with(expected_message)
    
    @patch('modules.logger.log')
    def test_log_warning_with_icon(self, mock_log, logger_manager):
        """아이콘 포함 경고 로그 테스트"""
        logger_manager.log_warning_with_icon("주의사항")
        expected_message = "[TestModule] ⚠️ 주의사항"
        mock_log.warning.assert_called_once_with(expected_message)
    
    @patch('modules.logger.log')
    def test_log_error_with_icon(self, mock_log, logger_manager):
        """아이콘 포함 오류 로그 테스트"""
        logger_manager.log_error_with_icon("오류 발생")
        expected_message = "[TestModule] ❌ 오류 발생"
        mock_log.error.assert_called_once_with(expected_message)
    
    def test_logger_without_module_name(self):
        """모듈명 없는 로거 테스트"""
        logger = LoggerManager()
        assert logger.module_name == "module"
    
    @patch('modules.logger.log')
    def test_logger_without_module_name_logging(self, mock_log):
        """모듈명 없는 로거의 로깅 테스트"""
        logger = LoggerManager()
        logger.info("테스트 메시지")
        mock_log.info.assert_called_once_with("[module] 테스트 메시지")
    
    def test_get_global_logger(self):
        """전역 로거 반환 테스트"""
        global_logger = LoggerManager.get_global_logger()
        assert global_logger is not None
    
    def test_file_mode_setting(self):
        """파일 모드 설정 테스트"""
        logger = LoggerManager(file_mode="a")
        assert logger.file_mode == "a"