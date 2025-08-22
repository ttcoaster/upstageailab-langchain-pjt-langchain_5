import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferWindowMemory
from langchain_upstage import ChatUpstage
from langchain_upstage import UpstageEmbeddings
 
# 현재 스크립트 위치를 작업 디렉토리로 설정
from pathlib import Path
script_dir = Path(__file__).parent.absolute()
os.chdir(script_dir)

import sys; sys.path.append('../');
import modules.logger as log
from vector_store import VectorStoreManager

# 환경변수 로드
load_dotenv()
log.info("환경변수가 성공적으로 로드되었습니다.")


# 단계 1-3: 임베딩(Embedding) 생성
log.info("임베딩 모델을 초기화하는 중...")
embeddings = UpstageEmbeddings(
    api_key=os.getenv("UPSTAGE_API_KEY"),
    model="embedding-query"
)
log.info("임베딩 모델이 성공적으로 초기화되었습니다.")

# 단계 4: 벡터스토어 관리자 생성 및 벡터스토어 로드/생성
log.info("벡터스토어 관리자를 초기화하는 중...")
vector_manager = VectorStoreManager(
    pdf_dir="../../data/pdf",
    vectorstore_dir="../../data/vectorstore", 
    embeddings=embeddings,
    chunk_size=1000,
    chunk_overlap=50
)

# 벡터스토어 가져오기 (기존 로드 또는 새로 생성, 증분 업데이트 자동 처리)
vectorstore = vector_manager.get_or_create_vectorstore()

if vectorstore is None:
    log.error("벡터스토어를 생성하거나 로드할 수 없습니다.")
    exit(1)

# 단계 5: 검색기(Retriever) 생성
retriever = vectorstore.as_retriever()
log.info("벡터 데이터베이스와 검색기가 성공적으로 생성되었습니다.")

# 단계 6: 프롬프트 생성(Create Prompt)
# 메모리 기능을 포함한 프롬프트를 생성합니다.
prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an assistant for question-answering tasks. 
Use the following pieces of retrieved context to answer the question. 
If you don't know the answer, just say that you don't know. 
Answer in Korean.

Context: {context}"""),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{question}")
])
log.info("프롬프트 템플릿이 생성되었습니다.")

# 단계 7: 언어모델(LLM) 생성
log.info("언어모델을 초기화하는 중...")
llm = ChatUpstage(
    api_key=os.getenv("UPSTAGE_API_KEY"),
    model="solar-pro2",
    reasoning_effort="high"
)
log.info("언어모델이 성공적으로 초기화되었습니다.")

# 단계 7-1: 메모리 생성 (최근 3개 대화 저장)
memory = ConversationBufferWindowMemory(
    k=3,  # 최근 3개 대화만 기억
    memory_key="chat_history",
    return_messages=True,
    output_key="answer"
)


# 단계 8: 체인(Chain) 생성 - 메모리 기능 포함
def create_rag_chain_with_memory():
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    # RAG 체인과 메모리를 결합하는 함수
    def rag_with_memory(question):
        # 검색된 문서 가져오기
        docs = retriever.invoke(question)
        context = format_docs(docs)
        
        # 메모리에서 채팅 기록 가져오기
        chat_history = memory.chat_memory.messages
        
        # 프롬프트에 필요한 변수들 준비
        formatted_prompt = prompt.format_messages(
            context=context,
            chat_history=chat_history,
            question=question
        )
        
        # LLM 호출
        response = llm.invoke(formatted_prompt)
        
        # 메모리에 대화 저장
        memory.save_context({"input": question}, {"answer": response.content})
        
        return response.content
    
    return rag_with_memory

# 메모리 기능이 포함된 RAG 체인 생성
chain = create_rag_chain_with_memory()
log.info("RAG 체인이 성공적으로 생성되었습니다.")

# 체인 실행(Run Chain) - 메모리 기능 테스트
# 5개 질문으로 메모리 기능을 테스트합니다.

questions = [
    "삼성전자가 자체 개발한 AI의 이름은?",
    "AI 브리프는 언제 발행되나요?", 
    "문서의 주요 내용을 요약해주세요",
    "네 번째 질문이야. 처음에 내가 뭘 물어봤는지 기억해?",
    "첫 번째 질문과 두 번째 질문을 기억하고 있어?"
]

log.info("=== LangChain 메모리 기능 테스트 ===")
log.info(f"메모리 설정: 최근 {memory.k}개 대화 기억\n")

for i, question in enumerate(questions, 1):
    log.info(f"🔵 질문 {i}: {question}")
    
    # 질문 실행
    response = chain(question)
    log.info(f"💬 답변: {response}")
    
    # 현재 메모리 상태 출력 (디버깅용)
    log.info(f"📝 현재 메모리에 저장된 대화 수: {len(memory.chat_memory.messages) // 2}")
    log.info("-" * 80)