"""
문서 수집 기능 모듈

이 모듈은 Knowledge Base 구축을 위한 문서 데이터를 수집하고 처리하는 기능을 제공합니다.

주요 기능:
1. PDF 파일 로딩 및 처리
2. 디렉토리 스캔 및 파일 수집
3. 문서 메타데이터 추출
4. 텍스트 분할 및 전처리

Author: AI Assistant
Date: 2025-08-22
"""

import os
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
from datetime import datetime

from langchain_community.document_loaders import DirectoryLoader, PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from .logger import LoggerManager


class CrawlerManager:
    """문서 수집 관리 클래스"""
    
    def __init__(self, 
                 base_directory: str = "../data/pdf",
                 chunk_size: int = 1000,
                 chunk_overlap: int = 50,
                 supported_extensions: List[str] = None):
        """
        CrawlerManager 초기화
        
        Args:
            base_directory (str): 기본 문서 디렉토리
            chunk_size (int): 텍스트 분할 크기
            chunk_overlap (int): 텍스트 분할 중복 크기
            supported_extensions (List[str], optional): 지원하는 파일 확장자
        """
        self.logger = LoggerManager("Crawler")
        self.base_directory = base_directory
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.supported_extensions = supported_extensions or ['.pdf']
        
        # 텍스트 분할기 초기화
        self._init_text_splitter()
        
        self.logger.log_success("Crawler Manager 초기화 완료")
    
    def _init_text_splitter(self):
        """텍스트 분할기 초기화"""
        try:
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                separators=["\n\n", "\n", " ", ""]
            )
            self.logger.log_step("텍스트 분할기 초기화", 
                               f"청크 크기: {self.chunk_size}, 중복: {self.chunk_overlap}")
        except Exception as e:
            self.logger.log_error("텍스트 분할기 초기화", e)
            raise
    
    def scan_directory(self, directory: str = None) -> List[Dict]:
        """
        디렉토리 스캔하여 파일 정보 수집
        
        Args:
            directory (str, optional): 스캔할 디렉토리. None이면 기본 디렉토리 사용
            
        Returns:
            List[Dict]: 파일 정보 리스트
        """
        scan_dir = directory or self.base_directory
        self.logger.log_function_start("scan_directory", directory=scan_dir)
        
        try:
            scan_path = Path(scan_dir)
            if not scan_path.exists():
                self.logger.log_warning_with_icon(f"디렉토리가 존재하지 않습니다: {scan_dir}")
                return []
            
            file_info_list = []
            
            for ext in self.supported_extensions:
                pattern = f"**/*{ext}"
                files = list(scan_path.rglob(pattern))
                
                for file_path in files:
                    if file_path.is_file():
                        stat = file_path.stat()
                        file_info = {
                            "absolute_path": str(file_path),
                            "relative_path": str(file_path.relative_to(scan_path)),
                            "filename": file_path.name,
                            "extension": file_path.suffix,
                            "size": stat.st_size,
                            "modified_time": datetime.fromtimestamp(stat.st_mtime),
                            "created_time": datetime.fromtimestamp(stat.st_ctime)
                        }
                        file_info_list.append(file_info)
            
            self.logger.log_function_end("scan_directory", 
                                       f"{len(file_info_list)}개 파일 발견")
            return file_info_list
            
        except Exception as e:
            self.logger.log_error("scan_directory", e)
            return []
    
    def load_single_pdf(self, file_path: str) -> List[Document]:
        """
        단일 PDF 파일 로드
        
        Args:
            file_path (str): PDF 파일 경로
            
        Returns:
            List[Document]: 로드된 문서 리스트
        """
        self.logger.log_function_start("load_single_pdf", file_path=file_path)
        
        try:
            if not os.path.exists(file_path):
                self.logger.log_warning_with_icon(f"파일이 존재하지 않습니다: {file_path}")
                return []
            
            loader = PyMuPDFLoader(file_path)
            documents = loader.load()
            
            # 메타데이터 보강
            for doc in documents:
                doc.metadata.update({
                    "source_file": os.path.basename(file_path),
                    "file_size": os.path.getsize(file_path),
                    "loaded_at": datetime.now().isoformat()
                })
            
            self.logger.log_function_end("load_single_pdf", 
                                       f"{len(documents)}개 페이지 로드")
            return documents
            
        except Exception as e:
            self.logger.log_error("load_single_pdf", e)
            return []
    
    def load_multiple_pdfs(self, file_paths: List[str]) -> List[Document]:
        """
        여러 PDF 파일 로드
        
        Args:
            file_paths (List[str]): PDF 파일 경로 리스트
            
        Returns:
            List[Document]: 로드된 모든 문서 리스트
        """
        self.logger.log_function_start("load_multiple_pdfs", count=len(file_paths))
        
        all_documents = []
        successful_files = 0
        
        for file_path in file_paths:
            documents = self.load_single_pdf(file_path)
            if documents:
                all_documents.extend(documents)
                successful_files += 1
        
        self.logger.log_function_end("load_multiple_pdfs", 
                                   f"{successful_files}/{len(file_paths)}개 파일, "
                                   f"{len(all_documents)}개 문서 로드")
        return all_documents
    
    def load_directory(self, directory: str = None, pattern: str = "**/*.pdf") -> List[Document]:
        """
        디렉토리에서 모든 파일 로드
        
        Args:
            directory (str, optional): 로드할 디렉토리. None이면 기본 디렉토리 사용
            pattern (str): 파일 패턴
            
        Returns:
            List[Document]: 로드된 모든 문서 리스트
        """
        load_dir = directory or self.base_directory
        self.logger.log_function_start("load_directory", directory=load_dir, pattern=pattern)
        
        try:
            loader = DirectoryLoader(
                load_dir,
                glob=pattern,
                loader_cls=PyMuPDFLoader,
                show_progress=True,
                use_multithreading=True
            )
            
            documents = loader.load()
            
            # 메타데이터 보강
            for doc in documents:
                source_path = doc.metadata.get('source', '')
                if source_path:
                    doc.metadata.update({
                        "source_file": os.path.basename(source_path),
                        "file_size": os.path.getsize(source_path) if os.path.exists(source_path) else 0,
                        "loaded_at": datetime.now().isoformat()
                    })
            
            self.logger.log_function_end("load_directory", 
                                       f"{len(documents)}개 문서 로드")
            return documents
            
        except Exception as e:
            self.logger.log_error("load_directory", e)
            return []
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        문서 리스트를 텍스트 분할기로 분할
        
        Args:
            documents (List[Document]): 원본 문서 리스트
            
        Returns:
            List[Document]: 분할된 문서 리스트
        """
        self.logger.log_function_start("split_documents", count=len(documents))
        
        try:
            split_docs = self.text_splitter.split_documents(documents)
            
            # 분할된 문서에 추가 메타데이터
            for i, doc in enumerate(split_docs):
                doc.metadata.update({
                    "chunk_id": i,
                    "chunk_size": len(doc.page_content),
                    "split_at": datetime.now().isoformat()
                })
            
            self.logger.log_function_end("split_documents", 
                                       f"{len(documents)} → {len(split_docs)}개 청크")
            return split_docs
            
        except Exception as e:
            self.logger.log_error("split_documents", e)
            return documents  # 실패 시 원본 반환
    
    def extract_metadata(self, documents: List[Document]) -> Dict[str, Any]:
        """
        문서 리스트에서 메타데이터 추출 및 통계 생성
        
        Args:
            documents (List[Document]): 문서 리스트
            
        Returns:
            Dict[str, Any]: 메타데이터 통계
        """
        if not documents:
            return {}
        
        # 기본 통계
        total_docs = len(documents)
        total_chars = sum(len(doc.page_content) for doc in documents)
        
        # 소스 파일별 통계
        source_stats = {}
        for doc in documents:
            source = doc.metadata.get('source_file', doc.metadata.get('source', 'Unknown'))
            if source not in source_stats:
                source_stats[source] = {"count": 0, "chars": 0}
            source_stats[source]["count"] += 1
            source_stats[source]["chars"] += len(doc.page_content)
        
        # 페이지별 통계 (있는 경우)
        page_count = len(set(doc.metadata.get('page', -1) for doc in documents if 'page' in doc.metadata))
        
        metadata_summary = {
            "total_documents": total_docs,
            "total_characters": total_chars,
            "average_doc_length": total_chars // total_docs if total_docs > 0 else 0,
            "unique_sources": len(source_stats),
            "unique_pages": page_count if page_count > 0 else None,
            "source_statistics": source_stats,
            "extracted_at": datetime.now().isoformat()
        }
        
        self.logger.log_step("메타데이터 추출", 
                           f"{total_docs}개 문서, {len(source_stats)}개 소스")
        
        return metadata_summary
    
    def process_documents_pipeline(self, 
                                 directory: str = None,
                                 pattern: str = "**/*.pdf") -> Tuple[List[Document], Dict[str, Any]]:
        """
        전체 문서 처리 파이프라인
        
        Args:
            directory (str, optional): 처리할 디렉토리
            pattern (str): 파일 패턴
            
        Returns:
            Tuple[List[Document], Dict[str, Any]]: (분할된 문서 리스트, 메타데이터)
        """
        self.logger.log_function_start("process_documents_pipeline")
        
        try:
            # 1. 문서 로드
            documents = self.load_directory(directory, pattern)
            if not documents:
                self.logger.log_warning_with_icon("로드된 문서가 없습니다.")
                return [], {}
            
            # 2. 메타데이터 추출 (분할 전)
            metadata = self.extract_metadata(documents)
            
            # 3. 문서 분할
            split_docs = self.split_documents(documents)
            
            # 4. 최종 메타데이터 업데이트
            metadata.update({
                "chunks_created": len(split_docs),
                "split_ratio": len(split_docs) / len(documents) if documents else 0
            })
            
            self.logger.log_function_end("process_documents_pipeline", 
                                       f"{len(documents)} → {len(split_docs)}개 청크 생성")
            
            return split_docs, metadata
            
        except Exception as e:
            self.logger.log_error("process_documents_pipeline", e)
            return [], {}
    
    def update_text_splitter_settings(self, 
                                    chunk_size: int = None,
                                    chunk_overlap: int = None):
        """
        텍스트 분할기 설정 업데이트
        
        Args:
            chunk_size (int, optional): 새 청크 크기
            chunk_overlap (int, optional): 새 중복 크기
        """
        updated = False
        
        if chunk_size and chunk_size != self.chunk_size:
            self.chunk_size = chunk_size
            updated = True
            self.logger.log_step("청크 크기 변경", f"새 크기: {chunk_size}")
        
        if chunk_overlap is not None and chunk_overlap != self.chunk_overlap:
            self.chunk_overlap = chunk_overlap
            updated = True
            self.logger.log_step("청크 중복 변경", f"새 중복: {chunk_overlap}")
        
        # 설정이 변경되었으면 분할기 재초기화
        if updated:
            self._init_text_splitter()
    
    def get_crawler_info(self) -> Dict[str, Any]:
        """현재 크롤러 설정 정보 반환"""
        return {
            "base_directory": self.base_directory,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "supported_extensions": self.supported_extensions,
            "directory_exists": os.path.exists(self.base_directory)
        }
    
    def validate_directory(self, directory: str = None) -> bool:
        """
        디렉토리 유효성 검사
        
        Args:
            directory (str, optional): 검사할 디렉토리
            
        Returns:
            bool: 유효성 검사 결과
        """
        check_dir = directory or self.base_directory
        
        try:
            path = Path(check_dir)
            if not path.exists():
                self.logger.log_warning_with_icon(f"디렉토리가 존재하지 않습니다: {check_dir}")
                return False
            
            if not path.is_dir():
                self.logger.log_warning_with_icon(f"경로가 디렉토리가 아닙니다: {check_dir}")
                return False
            
            # 읽기 권한 확인
            if not os.access(check_dir, os.R_OK):
                self.logger.log_warning_with_icon(f"디렉토리 읽기 권한이 없습니다: {check_dir}")
                return False
            
            self.logger.log_success(f"디렉토리 유효성 검사 통과: {check_dir}")
            return True
            
        except Exception as e:
            self.logger.log_error("validate_directory", e)
            return False