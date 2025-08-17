import json
import pytest
import numpy as np
from unittest.mock import AsyncMock, MagicMock, patch, mock_open
from pathlib import Path

from src.rag import to_embeddings, build_index, load_index, retrieve, INDEX_PATH, CONTENT_PATH


class TestToEmbeddings:
    @pytest.mark.asyncio
    @patch('src.rag.asyncio.to_thread')
    @patch('src.rag._model')
    async def test_to_embeddings(self, mock_model, mock_to_thread):
        """Проверяем создание эмбеддингов"""
        texts = ["Первый текст", "Второй текст"]
        mock_embeddings = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=np.float32)
        
        mock_to_thread.return_value = mock_embeddings
        
        result = await to_embeddings(texts)
        
        mock_to_thread.assert_called_once_with(
            mock_model.encode,
            texts,
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=True
        )
        
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float32
        np.testing.assert_array_equal(result, mock_embeddings)

    @pytest.mark.asyncio
    @patch('src.rag.asyncio.to_thread')
    async def test_to_embeddings_empty_list(self, mock_to_thread):
        """Проверяем обработку пустого списка"""
        texts = []
        mock_embeddings = np.array([], dtype=np.float32).reshape(0, 384)  # Стандартный размер
        
        mock_to_thread.return_value = mock_embeddings
        
        result = await to_embeddings(texts)
        
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float32
        assert result.shape[0] == 0


class TestBuildIndex:
    @pytest.mark.asyncio
    @patch('src.rag.faiss')
    @patch('src.rag.to_embeddings')
    async def test_build_index(self, mock_to_embeddings, mock_faiss):
        """Проверяем построение индекса"""
        mock_content = [
            {"url": "test1.com", "text": "Первый текст"},
            {"url": "test2.com", "text": "Второй текст"}
        ]
        
        mock_embeddings = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
        mock_to_embeddings.return_value = mock_embeddings
        
        mock_index = MagicMock()
        mock_faiss.IndexFlatIP.return_value = mock_index
        
        mock_content_path = MagicMock(spec=Path)
        mock_content_path.read_text.return_value = json.dumps(mock_content)
        
        await build_index(mock_content_path)
        
        mock_to_embeddings.assert_called_once_with(["Первый текст", "Второй текст"])
        mock_faiss.IndexFlatIP.assert_called_once_with(2)  # размерность эмбеддингов
        mock_index.add.assert_called_once_with(mock_embeddings)
        mock_faiss.write_index.assert_called_once_with(mock_index, str(INDEX_PATH))

    @pytest.mark.asyncio
    @patch('src.rag.faiss')
    @patch('src.rag.to_embeddings')
    async def test_build_index_empty_content(self, mock_to_embeddings, mock_faiss):
        """Проверяем построение индекса с пустым контентом"""
        mock_content = []
        mock_embeddings = np.array([], dtype=np.float32).reshape(0, 384)
        mock_to_embeddings.return_value = mock_embeddings
        
        mock_index = MagicMock()
        mock_faiss.IndexFlatIP.return_value = mock_index
        
        mock_content_path = MagicMock(spec=Path)
        mock_content_path.read_text.return_value = json.dumps(mock_content)
        
        await build_index(mock_content_path)
        
        mock_to_embeddings.assert_called_once_with([])
        mock_faiss.IndexFlatIP.assert_called_once_with(384)
        mock_index.add.assert_called_once_with(mock_embeddings)


class TestLoadIndex:
    @pytest.mark.asyncio
    @patch('src.rag.asyncio.to_thread')
    @patch('src.rag.faiss.read_index')
    async def test_load_index(self, mock_read_index, mock_to_thread):
        """Проверяем загрузку индекса"""
        mock_index = MagicMock()
        mock_to_thread.return_value = mock_index
        
        mock_index_path = MagicMock(spec=Path)
        
        result = await load_index(mock_index_path)
        
        mock_to_thread.assert_called_once_with(mock_read_index, str(mock_index_path))
        assert result is mock_index

    @pytest.mark.asyncio
    @patch('src.rag.asyncio.to_thread')
    async def test_load_index_default_path(self, mock_to_thread):
        """Проверяем загрузку индекса с путем по умолчанию"""
        mock_index = MagicMock()
        mock_to_thread.return_value = mock_index
        
        result = await load_index()
        
        mock_to_thread.assert_called_once()
        call_args = mock_to_thread.call_args[0]
        assert str(INDEX_PATH) in call_args[1]


class TestRetrieve:
    @pytest.mark.asyncio
    @patch('src.rag.to_embeddings')
    @patch('src.rag.asyncio.to_thread')
    async def test_retrieve_success(self, mock_to_thread, mock_to_embeddings):
        """Проверяем успешный поиск документов"""
        mock_index = MagicMock()
        mock_content = [
            {"url": "test1.com", "text": "Первый документ"},
            {"url": "test2.com", "text": "Второй документ"},
            {"url": "test3.com", "text": "Третий документ"}
        ]
        
        query = "тестовый запрос"
        query_emb = np.array([[0.1, 0.2, 0.3]], dtype=np.float32)
        mock_to_embeddings.return_value = query_emb
        
        scores = np.array([[0.8, 0.6]])
        ids = np.array([[0, 1]])
        mock_to_thread.return_value = (scores, ids)
        
        result = await retrieve(mock_index, mock_content, query, top_k=2, min_score=0.5)
        
        mock_to_embeddings.assert_called_once_with([query])
        mock_to_thread.assert_called_once_with(mock_index.search, query_emb, 2)
        
        expected = [
            {"url": "test1.com", "text": "Первый документ"},
            {"url": "test2.com", "text": "Второй документ"}
        ]
        assert result == expected

    @pytest.mark.asyncio
    @patch('src.rag.to_embeddings')
    @patch('src.rag.asyncio.to_thread')
    async def test_retrieve_with_min_score_filter(self, mock_to_thread, mock_to_embeddings):
        """Проверяем фильтрацию по минимальному скору"""
        mock_index = MagicMock()
        mock_content = [
            {"url": "test1.com", "text": "Первый документ"},
            {"url": "test2.com", "text": "Второй документ"},
            {"url": "test3.com", "text": "Третий документ"}
        ]
        
        query = "тестовый запрос"
        query_emb = np.array([[0.1, 0.2, 0.3]], dtype=np.float32)
        mock_to_embeddings.return_value = query_emb
        
        scores = np.array([[0.8, 0.3]])
        ids = np.array([[0, 1]])
        mock_to_thread.return_value = (scores, ids)
        
        result = await retrieve(mock_index, mock_content, query, top_k=2, min_score=0.5)
        
        expected = [
            {"url": "test1.com", "text": "Первый документ"}
        ]
        assert result == expected

    @pytest.mark.asyncio
    @patch('src.rag.to_embeddings')
    @patch('src.rag.asyncio.to_thread')
    async def test_retrieve_no_min_score(self, mock_to_thread, mock_to_embeddings):
        """Проверяем поиск без ограничения по скору"""
        mock_index = MagicMock()
        mock_content = [
            {"url": "test1.com", "text": "Первый документ"},
            {"url": "test2.com", "text": "Второй документ"}
        ]
        
        query = "тестовый запрос"
        query_emb = np.array([[0.1, 0.2, 0.3]], dtype=np.float32)
        mock_to_embeddings.return_value = query_emb
        
        scores = np.array([[0.1, 0.05]])
        ids = np.array([[0, 1]])
        mock_to_thread.return_value = (scores, ids)
        
        result = await retrieve(mock_index, mock_content, query, top_k=2, min_score=None)
        
        expected = [
            {"url": "test1.com", "text": "Первый документ"},
            {"url": "test2.com", "text": "Второй документ"}
        ]
        assert result == expected

    @pytest.mark.asyncio
    @patch('src.rag.to_embeddings')
    @patch('src.rag.asyncio.to_thread')
    async def test_retrieve_empty_results(self, mock_to_thread, mock_to_embeddings):
        """Проверяем случай без результатов"""
        mock_index = MagicMock()
        mock_content = [
            {"url": "test1.com", "text": "Первый документ"}
        ]
        
        query = "тестовый запрос"
        query_emb = np.array([[0.1, 0.2, 0.3]], dtype=np.float32)
        mock_to_embeddings.return_value = query_emb
        
        scores = np.array([[0.1]])
        ids = np.array([[0]])
        mock_to_thread.return_value = (scores, ids)
        
        result = await retrieve(mock_index, mock_content, query, top_k=1, min_score=0.5)
        
        assert result == []
