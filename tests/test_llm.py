import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch, mock_open
from pathlib import Path

from src.llm import LLMClient, base_prompt


class TestLLMClient:
    @patch('src.llm.rag')
    @patch('src.llm.config')
    @patch('src.llm.OpenAI')
    def test_init(self, mock_openai, mock_config, mock_rag):
        """Проверяем инициализацию LLMClient"""
        mock_config.llm_token = "test_token"
        mock_config.llm_url = "test_url"
        
        mock_content = [{"url": "test.com", "text": "test content"}]
        mock_rag.CONTENT_PATH.read_text.return_value = json.dumps(mock_content)
        
        client = LLMClient()
        
        mock_openai.assert_called_once_with(
            api_key="test_token",
            base_url="test_url"
        )
        assert client.content == mock_content

    @pytest.mark.asyncio
    @patch('src.llm.rag')
    async def test_init_async(self, mock_rag):
        """Проверяем асинхронную инициализацию"""
        mock_index = MagicMock()
        mock_rag.load_index = AsyncMock(return_value=mock_index)
        
        mock_content = [{"url": "test.com", "text": "test content"}]
        mock_rag.CONTENT_PATH.read_text.return_value = json.dumps(mock_content)
        
        client = LLMClient()
        await client.init()
        
        assert client.index is mock_index
        mock_rag.load_index.assert_called_once()

    @pytest.mark.asyncio
    @patch('src.llm.rag')
    async def test_build_prompt(self, mock_rag):
        """Проверяем построение промпта"""
        mock_content = [{"url": "test.com", "text": "test content"}]
        mock_rag.CONTENT_PATH.read_text.return_value = json.dumps(mock_content)
        
        client = LLMClient()
        client.index = MagicMock()
        client.content = [{"url": "test.com", "text": "test content"}]
        
        mock_rag.retrieve = AsyncMock(return_value=[
            {"url": "test1.com", "text": "content 1"},
            {"url": "test2.com", "text": "content 2"}
        ])
        
        user_question = "Тестовый вопрос"
        prompt = await client.build_prompt(user_question)
        
        mock_rag.retrieve.assert_called_once_with(
            client.index,
            client.content,
            user_question,
            top_k=2
        )
        
        expected_prompt = [
            {"role": "system", "content": base_prompt},
            {"role": "user", "content": user_question},
            {"role": "user", "content": "Контент из источника test1.com:\ncontent 1"},
            {"role": "user", "content": "Контент из источника test2.com:\ncontent 2"}
        ]
        
        assert prompt == expected_prompt

    @pytest.mark.asyncio
    async def test_generate_answer_success(self):
        """Проверяем успешную генерацию ответа"""
        client = LLMClient()
        
        client.build_prompt = AsyncMock(return_value=[{"role": "user", "content": "test"}])
        
        mock_response = MagicMock()
        mock_response.choices[0].message.content = '{"content": "Тестовый ответ", "urls": []}'
        
        client.client = MagicMock()
        client.client.chat.completions.create.return_value = mock_response
        
        result = await client.generate_answer("Тестовый вопрос")
        
        assert result == "Тестовый ответ"
        
        client.client.chat.completions.create.assert_called_once_with(
            model="deepseek-chat",
            messages=[{"role": "user", "content": "test"}],
            stream=False,
            temperature=0.5,
            top_p=0.8,
            max_tokens=300
        )

    @pytest.mark.asyncio
    async def test_generate_answer_with_urls(self):
        """Проверяем обработку ответа с URL-ами"""
        client = LLMClient()
        client.build_prompt = AsyncMock(return_value=[])
        
        mock_response = MagicMock()
        mock_response.choices[0].message.content = json.dumps({
            "content": "Ответ с ссылкой [1]",
            "urls": [{"1": "https://test.com"}]
        })
        
        client.client = MagicMock()
        client.client.chat.completions.create.return_value = mock_response
        
        result = await client.generate_answer("Тестовый вопрос")
        
        assert result == "Ответ с ссылкой [1](https://test.com)"

    @pytest.mark.asyncio
    async def test_generate_answer_json_fallback(self):
        """Проверяем fallback при некорректном JSON"""
        client = LLMClient()
        client.build_prompt = AsyncMock(return_value=[])
        
        mock_response = MagicMock()
        raw_content = "Некорректный JSON ответ"
        mock_response.choices[0].message.content = raw_content
        
        client.client = MagicMock()
        client.client.chat.completions.create.return_value = mock_response
        
        result = await client.generate_answer("Тестовый вопрос")
        
        assert result == raw_content

    @pytest.mark.asyncio
    async def test_generate_answer_with_code_blocks(self):
        """Проверяем обработку ответа с блоками кода"""
        client = LLMClient()
        client.build_prompt = AsyncMock(return_value=[])
        
        mock_response = MagicMock()
        mock_response.choices[0].message.content = '''```json
        {"content": "Ответ из блока кода", "urls": []}
        ```'''
        
        client.client = MagicMock()
        client.client.chat.completions.create.return_value = mock_response
        
        result = await client.generate_answer("Тестовый вопрос")
        
        assert result == "Ответ из блока кода"
