"""
CLI RAG 시스템

webui와 동일한 RAG 방식을 사용하여 하드코딩된 질문에 답변하는 CLI 도구입니다.

주요 기능:
- webui와 동일한 RAG 파이프라인 사용
- 하드코딩된 질문 처리
- 메모리 기반 대화 관리 (DB 저장 없음)
- 검색된 문서 소스 정보 표시
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from langchain_upstage import UpstageEmbeddings
from typing import List, Dict

# 현재 스크립트의 디렉토리를 sys.path에 추가 (이미 code 폴더 안에 있음)
script_dir = Path(__file__).parent.absolute()
if str(script_dir) not in sys.path:
    sys.path.insert(0, str(script_dir))

# 모듈 import
from modules import VectorStoreManager, LLMManager, RetrieverManager

# 환경변수 로드
load_dotenv(script_dir / '.env')


class SimpleChatHistory:
    """메모리 기반 간단한 대화 관리 클래스 (DB 저장 없음)"""
    
    def __init__(self):
        self.messages = []
    
    def add_user_message(self, content: str):
        """사용자 메시지 추가"""
        self.messages.append({"role": "user", "content": content})
    
    def add_ai_message(self, content: str):
        """AI 메시지 추가"""
        self.messages.append({"role": "assistant", "content": content})
    
    def get_chat_history_as_dicts(self) -> List[Dict]:
        """대화 히스토리를 딕셔너리 리스트로 반환"""
        return self.messages.copy()


def initialize_rag_system():
    """RAG 시스템 초기화 (webui와 동일한 방식)"""
    try:
        print("🚀 RAG 시스템 초기화 중...")
        
        # 임베딩 모델 초기화
        embeddings = UpstageEmbeddings(
            api_key=os.getenv("UPSTAGE_API_KEY"),
            model="embedding-query"
        )
        
        # 벡터스토어 관리자 초기화 (절대 경로 사용)
        current_dir = Path(__file__).parent.parent.absolute()  # code 폴더의 부모 폴더
        pdf_dir = str(current_dir / "data" / "pdf")
        vectorstore_dir = str(current_dir / "data" / "vectorstore")
        
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
            print("❌ 벡터스토어를 생성하거나 로드할 수 없습니다.")
            return None, None, None
        
        print("✅ 벡터스토어 초기화 완료")
        
        # LLM 관리자 초기화
        llm_manager = LLMManager()
        print("✅ LLM 관리자 초기화 완료")
        
        # 검색기 관리자 초기화
        retriever_manager = RetrieverManager(vectorstore=vectorstore)
        print("✅ 검색기 관리자 초기화 완료")
        
        return llm_manager, retriever_manager, vector_manager
    
    except Exception as e:
        print(f"❌ 시스템 초기화 오류: {str(e)}")
        return None, None, None


def process_question(question: str, llm_manager, retriever_manager, chat_history):
    """질문 처리 및 RAG 파이프라인 실행"""
    print(f"\n📝 질문: {question}")
    print("🔍 문서 검색 중...")
    
    try:
        # 문서 검색
        documents = retriever_manager.search_documents(question)
        context = retriever_manager.format_documents_for_context(documents)
        
        if not documents:
            print("⚠️  관련 문서를 찾을 수 없습니다.")
            return
        
        print(f"📄 {len(documents)}개의 관련 문서를 찾았습니다.")
        
        # 소스 정보 표시
        sources = retriever_manager.get_unique_sources(documents)
        if sources:
            print("📚 참조 문서:")
            for source in sources:
                print(f"   • {source}")
        
        # 채팅 히스토리 가져오기
        chat_history_dicts = chat_history.get_chat_history_as_dicts()
        
        print("\n🤖 AI 답변 생성 중...")
        
        # LLM 응답 생성
        response = llm_manager.generate_response(
            question=question,
            context=context,
            chat_history=chat_history_dicts
        )
        
        # 대화 히스토리에 추가
        chat_history.add_user_message(question)
        chat_history.add_ai_message(response)
        
        # 결과 출력
        print("\n" + "="*80)
        print("🤖 AI 답변:")
        print("="*80)
        print(response)
        print("="*80)
        
        return response
        
    except Exception as e:
        print(f"❌ 질문 처리 오류: {str(e)}")
        return None


def main():
    """메인 함수"""
    print("🎯 CLI RAG 시스템 시작")
    print("="*50)
    
    # RAG 시스템 초기화
    llm_manager, retriever_manager, vector_manager = initialize_rag_system()
    
    if not all([llm_manager, retriever_manager, vector_manager]):
        print("❌ 시스템 초기화에 실패했습니다.")
        return
    
    # 메모리 기반 대화 관리자 초기화
    chat_history = SimpleChatHistory()
    
    # 하드코딩된 질문
    hardcoded_question = "제과제빵에서 반죽 온도 관리 방법은 무엇인가요?"
    
    print("\n✅ 시스템 초기화 완료")
    print("💬 하드코딩된 질문으로 RAG 시스템 테스트를 시작합니다...")
    
    # 질문 처리
    response = process_question(hardcoded_question, llm_manager, retriever_manager, chat_history)
    
    if response:
        print(f"\n✅ 질문 처리 완료!")
        print(f"📊 현재 대화 기록: {len(chat_history.messages)}개 메시지")
    else:
        print("\n❌ 질문 처리에 실패했습니다.")
    
    print("\n🏁 CLI RAG 시스템 종료")


if __name__ == "__main__":
    main()