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
import utils.log_util as log
 
# 현재 스크립트 위치를 작업 디렉토리로 설정
import sys
from pathlib import Path
script_dir = Path(__file__).parent.absolute()
os.chdir(script_dir)

# 환경변수 로드
load_dotenv()
log.info("환경변수가 성공적으로 로드되었습니다.")


# 단계 1: 문서 로드(Load Documents)
log.info("PDF 문서를 로드하는 중...")
from langchain_community.document_loaders import DirectoryLoader

# data/pdf 폴더 및 모든 서브폴더의 PDF 파일을 읽어옴
pdf_dir = "../data/pdf"
loader = DirectoryLoader(
    path=pdf_dir,
    glob="**/*.pdf",
    loader_cls=PyMuPDFLoader, # PDF 파일을 읽어오는 로더 클래스
    show_progress=True, # 진행 상황을 표시할지 여부
    recursive=True, # 하위 폴더를 재귀적으로 검색할지 여부
    use_multithreading=True # 멀티스레딩을 사용할지 여부
)
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
# 프롬프트를 생성합니다.
prompt = PromptTemplate.from_template(
    """You are an assistant for question-answering tasks. 
Use the following pieces of retrieved context to answer the question. 
If you don't know the answer, just say that you don't know. 
Answer in Korean.

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
    reasoning_effort="high"
)
log.info("언어모델이 성공적으로 초기화되었습니다.")

# 단계 8: 체인(Chain) 생성
chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
log.info("RAG 체인이 성공적으로 생성되었습니다.")

# 체인 실행(Run Chain)
# 문서에 대한 질의를 입력하고, 답변을 출력합니다.
question = "단팥빵 레시피를 1/4로 양을 줄여서 표로 보여줘"
log.info(f"질문: {question}")
log.info("답변을 생성하는 중...")
response = chain.invoke(question)
log.info(f"답변: {response}")
