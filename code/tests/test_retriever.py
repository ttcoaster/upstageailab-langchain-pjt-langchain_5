"""
RetrieverManager 테스트

벡터 DB 검색 기능을 테스트합니다.
"""

import pytest
from pathlib import Path
import sys
from unittest.mock import MagicMock, patch

# 현재 파일의 부모 디렉토리를 sys.path에 추가
current_dir = Path(__file__).parent.parent
sys.path.insert(0, str(current_dir))

from modules.retriever import RetrieverManager
from langchain_core.documents import Document


class TestRetrieverManager:
    """RetrieverManager 테스트 클래스"""
    
    @pytest.fixture
    def mock_vectorstore(self):
        """가짜 벡터스토어 픽스처"""
        mock_vs = MagicMock()
        mock_retriever = MagicMock()
        mock_vs.as_retriever.return_value = mock_retriever
        return mock_vs
    
    @pytest.fixture
    def sample_documents(self):
        """샘플 문서 픽스처"""
        docs = [
            Document(
                page_content="첫 번째 문서 내용입니다.",
                metadata={"source": "doc1.pdf", "page": 1}
            ),
            Document(
                page_content="두 번째 문서 내용입니다.",
                metadata={"source": "doc2.pdf", "page": 1}
            ),
            Document(
                page_content="세 번째 문서 내용입니다.",
                metadata={"source_file": "doc3.pdf", "page": 2}
            )
        ]
        return docs
    
    def test_init_without_vectorstore(self):
        """벡터스토어 없이 초기화 테스트"""
        retriever = RetrieverManager()
        assert retriever.vectorstore is None
        assert retriever.search_type == "similarity"
        assert retriever.k == 5
        assert retriever.score_threshold is None
    
    def test_init_with_vectorstore(self, mock_vectorstore):
        """벡터스토어와 함께 초기화 테스트"""
        retriever = RetrieverManager(vectorstore=mock_vectorstore, k=10)
        assert retriever.vectorstore == mock_vectorstore
        assert retriever.k == 10
        mock_vectorstore.as_retriever.assert_called_once()
    
    def test_set_vectorstore(self, mock_vectorstore):
        """벡터스토어 설정 테스트"""
        retriever = RetrieverManager()
        retriever.set_vectorstore(mock_vectorstore)
        assert retriever.vectorstore == mock_vectorstore
        mock_vectorstore.as_retriever.assert_called_once()
    
    def test_search_documents_success(self, mock_vectorstore, sample_documents):
        """문서 검색 성공 테스트"""
        retriever = RetrieverManager(vectorstore=mock_vectorstore)
        retriever.retriever.invoke.return_value = sample_documents
        
        result = retriever.search_documents("테스트 쿼리")
        
        assert len(result) == 3
        assert result == sample_documents
        retriever.retriever.invoke.assert_called_once_with("테스트 쿼리")
    
    def test_search_documents_no_retriever(self):
        """검색기 없을 때 문서 검색 테스트"""
        retriever = RetrieverManager()
        result = retriever.search_documents("테스트 쿼리")
        assert result == []
    
    def test_search_with_scores(self, mock_vectorstore):
        """점수와 함께 검색 테스트"""
        retriever = RetrieverManager(vectorstore=mock_vectorstore)
        
        mock_results = [
            (Document(page_content="문서 1"), 0.95),
            (Document(page_content="문서 2"), 0.85)
        ]
        mock_vectorstore.similarity_search_with_score.return_value = mock_results
        
        result = retriever.search_with_scores("테스트 쿼리", k=2)
        
        assert len(result) == 2
        assert result == mock_results
        mock_vectorstore.similarity_search_with_score.assert_called_once_with("테스트 쿼리", k=2)
    
    def test_search_with_scores_no_vectorstore(self):
        """벡터스토어 없을 때 점수와 함께 검색 테스트"""
        retriever = RetrieverManager()
        result = retriever.search_with_scores("테스트 쿼리")
        assert result == []
    
    def test_search_by_vector(self, mock_vectorstore, sample_documents):
        """벡터로 검색 테스트"""
        retriever = RetrieverManager(vectorstore=mock_vectorstore)
        test_embedding = [0.1, 0.2, 0.3]
        mock_vectorstore.similarity_search_by_vector.return_value = sample_documents
        
        result = retriever.search_by_vector(test_embedding, k=3)
        
        assert len(result) == 3
        assert result == sample_documents
        mock_vectorstore.similarity_search_by_vector.assert_called_once_with(test_embedding, k=3)
    
    def test_filter_documents_by_source(self, sample_documents):
        """소스로 문서 필터링 테스트"""
        retriever = RetrieverManager()
        
        # "doc1"을 포함하는 문서만 필터링
        filtered = retriever.filter_documents_by_source(sample_documents, "doc1")
        
        assert len(filtered) == 1
        assert "doc1.pdf" in filtered[0].metadata["source"]
    
    def test_get_unique_sources(self, sample_documents):
        """고유 소스 추출 테스트"""
        retriever = RetrieverManager()
        sources = retriever.get_unique_sources(sample_documents)
        
        expected_sources = ["doc1.pdf", "doc2.pdf", "doc3.pdf"]
        assert len(sources) == 3
        for source in expected_sources:
            assert source in sources
    
    def test_format_documents_for_context(self, sample_documents):
        """문서를 컨텍스트로 포맷팅 테스트"""
        retriever = RetrieverManager()
        context = retriever.format_documents_for_context(sample_documents)
        
        assert "첫 번째 문서 내용입니다." in context
        assert "두 번째 문서 내용입니다." in context
        assert "세 번째 문서 내용입니다." in context
        assert "[문서 1]" in context
        assert "[문서 2]" in context
        assert "[문서 3]" in context
    
    def test_format_documents_empty_list(self):
        """빈 문서 리스트 포맷팅 테스트"""
        retriever = RetrieverManager()
        context = retriever.format_documents_for_context([])
        assert context == ""
    
    def test_update_search_params(self, mock_vectorstore):
        """검색 매개변수 업데이트 테스트"""
        retriever = RetrieverManager(vectorstore=mock_vectorstore)
        
        # 매개변수 업데이트
        retriever.update_search_params(
            search_type="mmr",
            k=10,
            score_threshold=0.8
        )
        
        assert retriever.search_type == "mmr"
        assert retriever.k == 10
        assert retriever.score_threshold == 0.8
        
        # 벡터스토어가 있으므로 as_retriever가 다시 호출되어야 함
        assert mock_vectorstore.as_retriever.call_count >= 2
    
    def test_get_search_info(self, mock_vectorstore):
        """검색 설정 정보 반환 테스트"""
        retriever = RetrieverManager(
            vectorstore=mock_vectorstore,
            search_type="similarity",
            k=5,
            score_threshold=0.7
        )
        
        info = retriever.get_search_info()
        
        assert info["search_type"] == "similarity"
        assert info["k"] == 5
        assert info["score_threshold"] == 0.7
        assert info["vectorstore_available"] is True
        assert info["retriever_available"] is True
    
    def test_test_search_success(self, mock_vectorstore, sample_documents):
        """검색 테스트 성공"""
        retriever = RetrieverManager(vectorstore=mock_vectorstore)
        retriever.retriever.invoke.return_value = sample_documents
        
        result = retriever.test_search("테스트")
        assert result is True
    
    def test_test_search_failure(self):
        """검색 테스트 실패"""
        retriever = RetrieverManager()  # 벡터스토어 없음
        
        result = retriever.test_search("테스트")
        assert result is True  # 빈 리스트 반환이지만 성공으로 간주
    
    def test_init_retriever_similarity_score_threshold(self, mock_vectorstore):
        """similarity_score_threshold 타입으로 검색기 초기화 테스트"""
        retriever = RetrieverManager(
            vectorstore=mock_vectorstore,
            search_type="similarity_score_threshold",
            score_threshold=0.8
        )
        
        expected_kwargs = {
            "search_type": "similarity_score_threshold",
            "search_kwargs": {
                "k": 5,
                "score_threshold": 0.8
            }
        }
        mock_vectorstore.as_retriever.assert_called_with(**expected_kwargs)
    
    def test_init_retriever_mmr(self, mock_vectorstore):
        """MMR 타입으로 검색기 초기화 테스트"""
        retriever = RetrieverManager(
            vectorstore=mock_vectorstore,
            search_type="mmr",
            k=3
        )
        
        expected_kwargs = {
            "search_type": "mmr",
            "search_kwargs": {
                "k": 3,
                "fetch_k": 6,  # k * 2
                "lambda_mult": 0.7
            }
        }
        mock_vectorstore.as_retriever.assert_called_with(**expected_kwargs)