from typing import Any, Awaitable, Callable

from aiogram import Bot, Dispatcher, Router, types, BaseMiddleware
from aiogram.filters import Command

from config import tg_token
from src import llm


router = Router()

class LLMClientMiddleware(BaseMiddleware):
    def __init__(self, llm_client: llm.LLMClient):
        self.llm_client = llm_client

    async def __call__(
        self,
        handler: Callable[[types.TelegramObject, dict[str, Any]], Awaitable[Any]],
        event: types.TelegramObject,
        data: dict[str, Any],
    ) -> Any:
        data["llm_client"] = self.llm_client
        return await handler(event, data)


@router.message(Command("start"))
async def start_command(message: types.Message):
    start_msg = "Привет! Я бот EORA.\nЗадавай мне вопросы о EORA, и я постараюсь помочь!"
    await message.answer(start_msg)


@router.message()
async def question_handler(message: types.Message, llm_client: llm.LLMClient):
    answer = await llm_client.generate_answer(
        message.text,
        max_tokens=300,
        temperature=0.5,
        top_p=0.8
    )
    try:
        await message.answer(answer, parse_mode="HTML")
    except Exception:
        await message.answer(answer)


async def create_bot(llm_client: llm.LLMClient) -> tuple[Bot, Dispatcher]:
    bot = Bot(token=tg_token)
    dp = Dispatcher()

    dp.update.outer_middleware(LLMClientMiddleware(llm_client))

    dp.include_router(router)
    
    return bot, dp