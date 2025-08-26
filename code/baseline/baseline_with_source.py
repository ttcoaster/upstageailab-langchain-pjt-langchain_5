import os
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_upstage import ChatUpstage
from langchain_upstage import UpstageEmbeddings

# 현재 스크립트 위치를 작업 디렉토리로 설정
from pathlib import Path
script_dir = Path(__file__).parent.absolute()
os.chdir(script_dir)

import log_util as log

# =========================
# [추가] 출처 문자열 포맷터
# =========================
def _format_sources(docs):
    """retriever가 고른 문서들의 파일명/페이지를 짧게 정리"""
    items = []
    for d in docs:
        meta = d.metadata or {}
        src = meta.get("source", "")
        page = meta.get("page", None)
        name = Path(src).name if src else "document"
        if page is not None:
            items.append(f"{name} p.{page + 1}")
        else:
            items.append(name)
    # 중복 제거(순서 보존)
    seen = set()
    uniq = []
    for it in items:
        if it not in seen:
            uniq.append(it)
            seen.add(it)
    return ", ".join(uniq)

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
# [변경] 상위 k 문서만 우선 활용하도록 튜닝
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
log.info("벡터 데이터베이스와 검색기가 성공적으로 생성되었습니다.")

# 단계 6: 프롬프트 생성(Create Prompt)
# 프롬프트를 생성합니다.
prompt = PromptTemplate.from_template(
    """You are an assistant for question-answering tasks. 
Use the following pieces of retrieved context to answer the question. 
If you don't know the answer, just say that you don't know. 
Answer in Korean. Keep it concise but include key facts.

#Question: 
{question} 
#Context: 
{context} 

#Answer:"""
)
log.info("프롬프트 템플릿이 생성되었습니다.")

# 단계 7: 언어모델(LLM) 생성
log.info("언어모델을 초기화하는 중...")
# 모델(LLM) 을 생성합니다.
# llm = ChatOpenAI(model_name="gpt-4o", temperature=0)
llm = ChatUpstage(
    api_key=os.getenv("UPSTAGE_API_KEY"),
    model="solar-pro2",
    reasoning_effort="high",
    temperature=0
)
log.info("언어모델이 성공적으로 초기화되었습니다.")

# 단계 8: 체인(Chain) 생성
# [변경] retriever를 체인 파이프라인에서 제거하고,
# 컨텍스트는 실행 시점에 직접 구성해 주입합니다.
chain = (
    prompt
    | llm
    | StrOutputParser()
)
log.info("RAG 체인이 성공적으로 생성되었습니다.")

# 체인 실행(Run Chain)
# 문서에 대한 질의를 입력하고, 답변을 출력합니다.
question = "삼성전자가 자체 개발한 AI 의 이름은?"
log.info(f"질문: {question}")

# [추가] 1) 먼저 관련 문서 가져오기
docs_top = retriever.invoke(question)

# [추가] 2) 컨텍스트 문자열 구성
context_text = "\n\n".join(d.page_content for d in docs_top)

# [변경] 3) 체인에 dict 형태로 question/context 직접 전달
log.info("답변을 생성하는 중...")
response = chain.invoke({"question": question, "context": context_text})

# [추가] 4) 출처 자동 덧붙이기
sources = _format_sources(docs_top)
if sources:
    response = f"{response}\n\n[출처] {sources}"

log.info(f"답변: {response}")
