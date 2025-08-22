"""
SQLManager 테스트

SQLite 데이터베이스 관리 기능을 테스트합니다.
"""

import pytest
import tempfile
import os
from pathlib import Path
import sys

# 현재 파일의 부모 디렉토리를 sys.path에 추가
current_dir = Path(__file__).parent.parent
sys.path.insert(0, str(current_dir))

from modules.sql import SQLManager


class TestSQLManager:
    """SQLManager 테스트 클래스"""
    
    @pytest.fixture
    def temp_db(self):
        """임시 데이터베이스 픽스처"""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
            temp_db_path = tmp.name
        
        yield temp_db_path
        
        # 테스트 후 정리
        if os.path.exists(temp_db_path):
            os.unlink(temp_db_path)
    
    @pytest.fixture
    def sql_manager(self, temp_db):
        """SQLManager 인스턴스 픽스처"""
        return SQLManager(db_path=temp_db)
    
    def test_create_conversation(self, sql_manager):
        """대화 생성 테스트"""
        session_id = sql_manager.create_conversation("테스트 대화")
        assert session_id is not None
        assert isinstance(session_id, str)
        assert len(session_id) > 0
    
    def test_get_conversation_id(self, sql_manager):
        """세션 ID로 대화 ID 조회 테스트"""
        session_id = sql_manager.create_conversation("테스트 대화")
        conversation_id = sql_manager.get_conversation_id(session_id)
        
        assert conversation_id is not None
        assert isinstance(conversation_id, int)
        assert conversation_id > 0
    
    def test_add_message(self, sql_manager):
        """메시지 추가 테스트"""
        session_id = sql_manager.create_conversation("테스트 대화")
        
        # 사용자 메시지 추가
        user_msg_id = sql_manager.add_message(session_id, "user", "안녕하세요")
        assert user_msg_id is not None
        assert isinstance(user_msg_id, int)
        
        # AI 메시지 추가
        ai_msg_id = sql_manager.add_message(session_id, "assistant", "안녕하세요! 무엇을 도와드릴까요?")
        assert ai_msg_id is not None
        assert isinstance(ai_msg_id, int)
        assert ai_msg_id != user_msg_id
    
    def test_get_messages(self, sql_manager):
        """메시지 조회 테스트"""
        session_id = sql_manager.create_conversation("테스트 대화")
        
        # 메시지 추가
        sql_manager.add_message(session_id, "user", "첫 번째 질문")
        sql_manager.add_message(session_id, "assistant", "첫 번째 답변")
        sql_manager.add_message(session_id, "user", "두 번째 질문")
        
        # 메시지 조회
        messages = sql_manager.get_messages(session_id)
        
        assert len(messages) == 3
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "첫 번째 질문"
        assert messages[1]["role"] == "assistant"
        assert messages[1]["content"] == "첫 번째 답변"
        assert messages[2]["role"] == "user"
        assert messages[2]["content"] == "두 번째 질문"
    
    def test_get_conversations(self, sql_manager):
        """대화 목록 조회 테스트"""
        # 여러 대화 생성
        session1 = sql_manager.create_conversation("첫 번째 대화")
        session2 = sql_manager.create_conversation("두 번째 대화")
        
        # 메시지 추가
        sql_manager.add_message(session1, "user", "질문1")
        sql_manager.add_message(session2, "user", "질문2")
        
        # 대화 목록 조회
        conversations = sql_manager.get_conversations()
        
        assert len(conversations) >= 2
        
        # 최신 순으로 정렬되는지 확인
        titles = [conv["title"] for conv in conversations]
        assert "첫 번째 대화" in titles
        assert "두 번째 대화" in titles
    
    def test_update_conversation_title(self, sql_manager):
        """대화 제목 업데이트 테스트"""
        session_id = sql_manager.create_conversation("원래 제목")
        
        # 제목 업데이트
        success = sql_manager.update_conversation_title(session_id, "새로운 제목")
        assert success is True
        
        # 업데이트된 제목 확인
        conversations = sql_manager.get_conversations()
        updated_conv = next((c for c in conversations if c["session_id"] == session_id), None)
        assert updated_conv is not None
        assert updated_conv["title"] == "새로운 제목"
    
    def test_delete_conversation(self, sql_manager):
        """대화 삭제 테스트"""
        session_id = sql_manager.create_conversation("삭제될 대화")
        sql_manager.add_message(session_id, "user", "테스트 메시지")
        
        # 대화 삭제
        success = sql_manager.delete_conversation(session_id)
        assert success is True
        
        # 삭제 확인
        conversation_id = sql_manager.get_conversation_id(session_id)
        assert conversation_id is None
        
        messages = sql_manager.get_messages(session_id)
        assert len(messages) == 0
    
    def test_get_recent_messages(self, sql_manager):
        """최근 메시지 조회 테스트"""
        session_id = sql_manager.create_conversation("테스트 대화")
        
        # 여러 메시지 추가
        for i in range(5):
            sql_manager.add_message(session_id, "user", f"질문 {i+1}")
            sql_manager.add_message(session_id, "assistant", f"답변 {i+1}")
        
        # 최근 4개 메시지 조회
        recent_messages = sql_manager.get_recent_messages(session_id, count=4)
        
        assert len(recent_messages) == 4
        assert recent_messages[-1][1] == "답변 5"  # 마지막 메시지
        assert recent_messages[-2][1] == "질문 5"  # 마지막에서 두 번째
    
    def test_invalid_session_id(self, sql_manager):
        """존재하지 않는 세션 ID 테스트"""
        fake_session_id = "non-existent-session"
        
        # 존재하지 않는 세션의 conversation_id는 None이어야 함
        conversation_id = sql_manager.get_conversation_id(fake_session_id)
        assert conversation_id is None
        
        # 존재하지 않는 세션에 메시지 추가 시 오류 발생
        with pytest.raises(ValueError):
            sql_manager.add_message(fake_session_id, "user", "테스트")
    
    def test_message_metadata(self, sql_manager):
        """메시지 메타데이터 테스트"""
        session_id = sql_manager.create_conversation("메타데이터 테스트")
        
        metadata = {"source": "test", "confidence": 0.95}
        message_id = sql_manager.add_message(
            session_id, "assistant", "메타데이터가 있는 메시지", metadata
        )
        
        assert message_id is not None
        
        messages = sql_manager.get_messages(session_id)
        assert len(messages) == 1
        assert messages[0]["metadata"] == metadata