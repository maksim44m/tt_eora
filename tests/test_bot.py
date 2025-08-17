import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from aiogram import types

from src.bot import create_bot, question_handler, start_command
from src.llm import LLMClient


class TestBotHandlers:
    @pytest.mark.asyncio
    async def test_start_command(self):
        """Проверяем корректный ответ на команду /start"""
        mock_message = AsyncMock(spec=types.Message)
        mock_message.answer = AsyncMock()
        
        await start_command(mock_message)
        
        expected_text = "Привет! Я бот EORA.\nЗадавай мне вопросы о EORA, и я постараюсь помочь!"
        mock_message.answer.assert_called_once_with(expected_text)

    @pytest.mark.asyncio
    async def test_question_handler_success(self):
        """Проверяем обработку вопроса с успешным ответом"""
        mock_message = AsyncMock(spec=types.Message)
        mock_message.text = "Тестовый вопрос"
        mock_message.answer = AsyncMock()
        
        mock_llm_client = AsyncMock(spec=LLMClient)
        mock_llm_client.generate_answer.return_value = "Тестовый ответ"
        
        await question_handler(mock_message, mock_llm_client)
        
        mock_llm_client.generate_answer.assert_called_once_with(
            "Тестовый вопрос",
            max_tokens=300,
            temperature=0.5,
            top_p=0.8
        )
        mock_message.answer.assert_called_once_with("Тестовый ответ", parse_mode="Markdown")

    @pytest.mark.asyncio
    async def test_question_handler_markdown_fallback(self):
        """Проверяем fallback при ошибке с Markdown"""
        mock_message = AsyncMock(spec=types.Message)
        mock_message.text = "Тестовый вопрос"
        
        mock_answer = AsyncMock()
        mock_answer.side_effect = [Exception("Markdown error"), None]
        mock_message.answer = mock_answer
        
        mock_llm_client = AsyncMock(spec=LLMClient)
        mock_llm_client.generate_answer.return_value = "Тестовый ответ"
        
        await question_handler(mock_message, mock_llm_client)
        
        assert mock_message.answer.call_count == 2
        mock_message.answer.assert_any_call("Тестовый ответ", parse_mode="Markdown")
        mock_message.answer.assert_any_call("Тестовый ответ")


class TestCreateBot:
    @pytest.mark.asyncio
    @patch('src.bot.Bot')
    @patch('src.bot.Dispatcher')
    async def test_create_bot(self, mock_dispatcher_class, mock_bot_class):
        """Проверяем создание бота и диспетчера"""
        mock_llm_client = MagicMock(spec=LLMClient)
        mock_bot = MagicMock()
        mock_dp = MagicMock()
        
        mock_bot_class.return_value = mock_bot
        mock_dispatcher_class.return_value = mock_dp
        
        with patch('src.bot.tg_token', 'test_token'):
            bot, dp = await create_bot(mock_llm_client)
        
        mock_bot_class.assert_called_once_with(token='test_token')
        mock_dp.update.outer_middleware.assert_called_once()
        mock_dp.include_router.assert_called_once()
        
        assert bot is mock_bot
        assert dp is mock_dp
