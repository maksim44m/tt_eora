import asyncio
from pathlib import Path

import config
from src import bot, llm, parser, rag

logger = config.logging.getLogger(__name__)


async def main():
    # 1. Создание content.json если его нет
    content_path = Path("data/content.json")
    if not content_path.exists():
        logger.info("Создаем content.json...")
        await parser.main()
    
    # 2. Создание RAG индекса если его нет
    index_path = Path("data/index.faiss")
    if not index_path.exists():
        logger.info("Создаем RAG индекс...")
        await rag.build_index()
    
    # 3. Инициализация LLM клиента
    llm_client = llm.LLMClient()
    await llm_client.init()
    
    # 4. Запуск Telegram бота
    logger.info("Запускаем Telegram бота...")
    bot_instance, dp = await bot.create_bot(llm_client)
    await dp.start_polling(bot_instance)


if __name__ == "__main__":
    asyncio.run(main())