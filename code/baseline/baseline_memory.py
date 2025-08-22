import os
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferWindowMemory
from langchain_upstage import ChatUpstage
from langchain_upstage import UpstageEmbeddings
 
# 현재 스크립트 위치를 작업 디렉토리로 설정
from pathlib import Path
script_dir = Path(__file__).parent.absolute()
os.chdir(script_dir)

import sys; sys.path.append('../'); import modules.logger as log

# 환경변수 로드
load_dotenv()
log.info("환경변수가 성공적으로 로드되었습니다.")


# 단계 1: 문서 로드(Load Documents)
log.info("PDF 문서를 로드하는 중...")
loader = PyMuPDFLoader("../../data/pdf/SPRI_AI_Brief_2023년12월호_F.pdf")
# loader = PyMuPDFLoader("../../data/pdf/4.단팥빵(비상스트레이트법).pdf")
docs = loader.load()
log.info(f"로드된 문서 수: {len(docs)}")
log.info(f"첫 번째 문서 미리보기: {docs[0].page_content[:200]}...")

# 단계 2: 문서 분할(Split Documents)
log.info("문서를 청크로 분할하는 중...")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
split_documents = text_splitter.split_documents(docs)
log.info(f"분할된 문서 청크 수: {len(split_documents)}")
log.info(f"첫 번째 청크 미리보기: {split_documents[0].page_content[:200]}...")

# 단계 3: 임베딩(Embedding) 생성
log.info("임베딩 모델을 초기화하는 중...")
# embeddings = OpenAIEmbeddings()
embeddings = UpstageEmbeddings(
    api_key=os.getenv("UPSTAGE_API_KEY"),
    model="embedding-query"
)
log.info("임베딩 모델이 성공적으로 초기화되었습니다.")

# 단계 4: DB 생성(Create DB) 및 저장
log.info("벡터 데이터베이스를 생성하는 중...")
# 벡터스토어를 생성합니다.
vectorstore = FAISS.from_documents(documents=split_documents, embedding=embeddings)

# 단계 5: 검색기(Retriever) 생성
# 문서에 포함되어 있는 정보를 검색하고 생성합니다.
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
# 모델(LLM) 을 생성합니다.
# llm = ChatOpenAI(model_name="gpt-4o", temperature=0)
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