"""
pytest 설정 파일

공통 fixture와 설정을 정의합니다.
"""

import pytest
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock


@pytest.fixture(scope="session")
def test_data_dir():
    """테스트 데이터 디렉토리 픽스처"""
    current_dir = Path(__file__).parent
    test_data_dir = current_dir / "test_data"
    test_data_dir.mkdir(exist_ok=True)
    return test_data_dir


@pytest.fixture
def temp_dir():
    """임시 디렉토리 픽스처"""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)


@pytest.fixture
def mock_upstage_api_key():
    """Upstage API 키 모킹"""
    original_key = os.environ.get("UPSTAGE_API_KEY")
    os.environ["UPSTAGE_API_KEY"] = "test_api_key"
    
    yield "test_api_key"
    
    # 원래 값 복원
    if original_key:
        os.environ["UPSTAGE_API_KEY"] = original_key
    else:
        os.environ.pop("UPSTAGE_API_KEY", None)


@pytest.fixture
def mock_embeddings():
    """가짜 임베딩 모델 픽스처"""
    mock_emb = MagicMock()
    mock_emb.embed_query.return_value = [0.1, 0.2, 0.3, 0.4, 0.5]
    mock_emb.embed_documents.return_value = [[0.1, 0.2, 0.3, 0.4, 0.5]]
    return mock_emb


@pytest.fixture
def mock_llm():
    """가짜 LLM 모델 픽스처"""
    mock_llm = MagicMock()
    mock_response = MagicMock()
    mock_response.content = "테스트 응답입니다."
    mock_llm.invoke.return_value = mock_response
    return mock_llm


# pytest 설정
def pytest_configure(config):
    """pytest 설정"""
    config.addinivalue_line(
        "markers", "integration: mark test as integration test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )