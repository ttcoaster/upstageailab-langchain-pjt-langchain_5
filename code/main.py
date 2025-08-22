"""
Streamlit Chat WebUI

LangChain 기반 RAG 시스템을 위한 Streamlit 웹 인터페이스입니다.

주요 기능:
- 채팅 인터페이스
- 대화 히스토리 관리
- 설정 패널
- 문서 검색 및 응답 생성

Author: AI Assistant
Date: 2025-08-22
"""

import os
import sys
import streamlit as st
from pathlib import Path
from dotenv import load_dotenv
from langchain_upstage import UpstageEmbeddings

# 현재 스크립트의 디렉토리를 sys.path에 추가 (chdir 대신)
script_dir = Path(__file__).parent.absolute()
if str(script_dir) not in sys.path:
    sys.path.insert(0, str(script_dir))

# 모듈 import
from modules import (
    SQLManager, LoggerManager, VectorStoreManager, 
    LLMManager, RetrieverManager, ChatHistoryManager, CrawlerManager
)

# 환경변수 로드 (스크립트 디렉토리 기준)
load_dotenv(script_dir / '.env')

# 페이지 설정
st.set_page_config(
    page_title="RAG Chat Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS 스타일링
st.markdown("""
<style>
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
        flex-direction: column;
    }
    .user-message {
        background-color: #f0f2f6;
        margin-left: 20%;
    }
    .assistant-message {
        background-color: #e8f4fd;
        margin-right: 20%;
    }
    .message-header {
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    .stApp > header {
        background-color: transparent;
    }
    .stApp {
        margin-top: -80px;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def initialize_system():
    """시스템 초기화 (캐시됨)"""
    try:
        # 임베딩 모델 초기화
        embeddings = UpstageEmbeddings(
            api_key=os.getenv("UPSTAGE_API_KEY"),
            model="embedding-query"
        )
        
        # 벡터스토어 관리자 초기화 (절대 경로 사용)
        current_dir = Path(__file__).parent.absolute()
        pdf_dir = str(current_dir.parent / "data" / "pdf")
        vectorstore_dir = str(current_dir.parent / "data" / "vectorstore")
        
        vector_manager = VectorStoreManager(
            pdf_dir=pdf_dir,
            vectorstore_dir=vectorstore_dir, 
            embeddings=embeddings,
            chunk_size=1000,
            chunk_overlap=50
        )
        
        # 벡터스토어 로드/생성
        vectorstore = vector_manager.get_or_create_vectorstore()
        
        if vectorstore is None:
            st.error("벡터스토어를 생성하거나 로드할 수 없습니다.")
            return None, None, None, None
        
        # LLM 관리자 초기화
        llm_manager = LLMManager()
        
        # 검색기 관리자 초기화
        retriever_manager = RetrieverManager(vectorstore=vectorstore)
        
        # SQL 관리자 초기화 (절대 경로 사용)
        db_path = str(current_dir.parent / "data" / "chat.db")
        sql_manager = SQLManager(db_path=db_path)
        
        return vector_manager, llm_manager, retriever_manager, sql_manager
    
    except Exception as e:
        st.error(f"시스템 초기화 오류: {str(e)}")
        return None, None, None, None


def initialize_session_state():
    """세션 상태 초기화"""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "current_session_id" not in st.session_state:
        st.session_state.current_session_id = None
    
    if "chat_history_manager" not in st.session_state:
        st.session_state.chat_history_manager = None
    
    if "show_sources" not in st.session_state:
        st.session_state.show_sources = False


def create_new_conversation(sql_manager):
    """새 대화 생성"""
    try:
        # 새 세션 생성
        session_id = sql_manager.create_conversation()
        
        # 채팅 히스토리 관리자 초기화
        chat_manager = ChatHistoryManager(
            session_id=session_id,
            sql_manager=sql_manager
        )
        
        # 세션 상태 업데이트
        st.session_state.current_session_id = session_id
        st.session_state.chat_history_manager = chat_manager
        st.session_state.messages = []
        
        st.success("새 대화가 시작되었습니다!")
        st.rerun()
        
    except Exception as e:
        st.error(f"새 대화 생성 오류: {str(e)}")


def load_conversation(session_id, sql_manager):
    """기존 대화 로드"""
    try:
        # 채팅 히스토리 관리자 초기화
        chat_manager = ChatHistoryManager(
            session_id=session_id,
            sql_manager=sql_manager
        )
        
        # 전체 대화 기록 가져오기
        messages = chat_manager.get_full_conversation_history()
        
        # 세션 상태 업데이트
        st.session_state.current_session_id = session_id
        st.session_state.chat_history_manager = chat_manager
        st.session_state.messages = messages
        
        st.success(f"대화를 불러왔습니다! ({len(messages)}개 메시지)")
        st.rerun()
        
    except Exception as e:
        st.error(f"대화 로드 오류: {str(e)}")


def render_sidebar(sql_manager):
    """사이드바 렌더링"""
    with st.sidebar:
        st.header("🗨️ 대화 관리")
        
        # 새 대화 버튼
        if st.button("🆕 새 대화 시작", use_container_width=True):
            create_new_conversation(sql_manager)
        
        st.divider()
        
        # 대화 목록
        st.subheader("📋 대화 기록")
        
        try:
            conversations = sql_manager.get_conversations(limit=20)
            
            if conversations:
                for conv in conversations:
                    session_id = conv["session_id"]
                    title = conv["title"]
                    updated_at = conv["updated_at"]
                    message_count = conv["message_count"]
                    
                    # 현재 선택된 대화 표시
                    is_current = (st.session_state.current_session_id == session_id)
                    button_label = f"{'🔵' if is_current else '⚪'} {title}"
                    
                    if st.button(
                        button_label, 
                        key=f"conv_{session_id}",
                        help=f"메시지: {message_count}개, 업데이트: {updated_at}",
                        use_container_width=True
                    ):
                        if not is_current:
                            load_conversation(session_id, sql_manager)
            else:
                st.info("저장된 대화가 없습니다.")
                
        except Exception as e:
            st.error(f"대화 목록 로드 오류: {str(e)}")
        
        st.divider()
        
        # 설정 패널
        st.subheader("⚙️ 설정")
        
        # 소스 표시 토글
        st.session_state.show_sources = st.checkbox(
            "검색된 문서 소스 표시", 
            value=st.session_state.show_sources
        )
        
        # 시스템 정보
        with st.expander("ℹ️ 시스템 정보"):
            if st.session_state.current_session_id:
                st.write(f"**현재 세션:** {st.session_state.current_session_id[:8]}...")
            
            if st.session_state.chat_history_manager:
                summary = st.session_state.chat_history_manager.get_conversation_summary()
                st.write(f"**전체 메시지:** {summary.get('total_messages', 0)}개")
                st.write(f"**메모리 메시지:** {summary.get('memory_messages', 0)}개")


def render_chat_interface(llm_manager, retriever_manager):
    """채팅 인터페이스 렌더링"""
    st.header("🤖 RAG Chat Assistant")
    
    # 세션이 없으면 안내 메시지
    if not st.session_state.current_session_id:
        st.info("👈 사이드바에서 '새 대화 시작'을 클릭하여 대화를 시작하세요.")
        return
    
    # 메시지 표시 영역
    chat_container = st.container()
    
    with chat_container:
        for message in st.session_state.messages:
            role = message["role"]
            content = message["content"]
            timestamp = message.get("timestamp", "")
            
            if role == "user":
                st.markdown(f"""
                <div class="chat-message user-message">
                    <div class="message-header">👤 사용자 {timestamp}</div>
                    <div>{content}</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="chat-message assistant-message">
                    <div class="message-header">🤖 AI 어시스턴트 {timestamp}</div>
                    <div>{content}</div>
                </div>
                """, unsafe_allow_html=True)
    
    # 사용자 입력
    user_input = st.chat_input("질문을 입력하세요...")
    
    if user_input and st.session_state.chat_history_manager:
        # 사용자 메시지 추가
        st.session_state.chat_history_manager.add_user_message(user_input)
        
        # 검색 수행
        with st.spinner("문서를 검색하고 답변을 생성하는 중..."):
            try:
                # 문서 검색
                documents = retriever_manager.search_documents(user_input)
                context = retriever_manager.format_documents_for_context(documents)
                
                # 채팅 히스토리 가져오기
                chat_history = st.session_state.chat_history_manager.get_chat_history_as_dicts()
                
                # LLM 응답 생성
                response = llm_manager.generate_response(
                    question=user_input,
                    context=context,
                    chat_history=chat_history
                )
                
                # AI 메시지 추가
                st.session_state.chat_history_manager.add_ai_message(response, user_input)
                
                # UI 메시지 리스트 업데이트
                st.session_state.messages = st.session_state.chat_history_manager.get_full_conversation_history()
                
                # 소스 정보 표시 (옵션)
                if st.session_state.show_sources and documents:
                    with st.expander(f"📄 검색된 문서 ({len(documents)}개)"):
                        sources = retriever_manager.get_unique_sources(documents)
                        for source in sources:
                            st.write(f"• {source}")
                
                st.rerun()
                
            except Exception as e:
                st.error(f"응답 생성 오류: {str(e)}")


def main():
    """메인 함수"""
    # 세션 상태 초기화
    initialize_session_state()
    
    # 시스템 초기화
    vector_manager, llm_manager, retriever_manager, sql_manager = initialize_system()
    
    if not all([vector_manager, llm_manager, retriever_manager, sql_manager]):
        st.error("시스템 초기화에 실패했습니다.")
        return
    
    # UI 렌더링
    render_sidebar(sql_manager)
    render_chat_interface(llm_manager, retriever_manager)


if __name__ == "__main__":
    main()