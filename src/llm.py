import json
from typing import Any

from openai import OpenAI

import config
from src import rag


logger = config.logging.getLogger(__name__)


base_prompt = """
    Ты — специалист по компании EORA (https://eora.ru/), ее продуктах, услугах, проектах и т.д.
    Твоя задача — отвечать на вопросы потенциальных клиентов о компании EORA.
    Ответы должны быть на русском языке.
    Ответы строятся на основе переданного контекста и вопроса пользователя.
    Информация в контексте может быть неупорядоченной.
    Никаких домыслов, общих рассуждений и сведений не подтвержденных источниками.
    
    Схема ответа:
    class BasePromptResult(BaseModel):
        content: str
        urls: list[dict[str, str]]

    Пример ответа на вопрос "Что вы можете сделать для ритейлеров?":
    {
        "content": "Например, мы делали бота для HR для Магнита [1], а ещё поиск по картинкам для KazanExpress [2]",
        "urls": [
            {"1": "https://eora.ru/cases/chat-boty/hr-bot-dlya-magnit-kotoriy-priglashaet-na-sobesedovanie"},
            {"2": "https://eora.ru/cases/kazanexpress-poisk-tovarov-po-foto"}
        ]
    }

    ВАЖНО:
    — Формат ответа JSON.
    — В "urls" включай ТОЛЬКО реально использованные номера и ссылки из "content", номера без дубликатов, в порядке возрастания, ссылки могут повторяться.
    — Если в контексте нет информации, то отвечай на основании сайта EORA (https://eora.ru/), в "urls" ничего не добавляй ("urls": []).
    — Никакого дополнительного текста или пояснений, только JSON.
    — Если ответишь не в формате JSON, то ты уволен.
    — Если ответишь в формате JSON, но он не распарсится, то ты уволен.
"""


class LLMClient:
    def __init__(self):
        self.client: OpenAI = OpenAI(
            api_key=config.llm_token, 
            base_url=config.llm_url
        )
        self.content: list[dict[str, Any]] = json.loads(
            rag.CONTENT_PATH.read_text(encoding="utf-8")
        )

    async def init(self):
        self.index = await rag.load_index()

    async def build_prompt(self, user_question: str):
        result_contents = await rag.retrieve(
            self.index,
            self.content,
            user_question,
            top_k=2
        )
        context = '\n'.join([content["text"] for content in result_contents])

        logger.info(f"question: {user_question}")
        logger.info(f'context: {context}')

        prompt = [
            {"role": "system", "content": base_prompt},
            {"role": "user", "content": user_question}
        ]

        for content in result_contents:
            prompt.append({
                "role": "user", 
                "content": f'Контент из источника {content["url"]}:\n{content["text"]}'
            })
            
        return prompt

    async def generate_answer(
        self,
        user_question: str,
        max_tokens: int = 300,
        temperature: float = 0.5,
        top_p: float = 0.8
    ) -> str:
        """
        user_question:
        max_tokens: максимальное количество токенов в ответе
        temperature: ближе к нулю - более детерминированные ответы, 1 (default) наиболее разнообразные
        top_p: Ограничение выбора токенов: 1=100% выборки, 0.5=50% выборки (больше фокуса)
        """

        answer = self.client.chat.completions.create(
            model="deepseek-chat",
            messages=await self.build_prompt(user_question),
            stream=False,
            temperature=round(temperature, 2),
            top_p=round(top_p, 2),
            max_tokens=max_tokens
        )
        logger.info(f'answer: {answer}')

        try:
            raw_content = answer.choices[0].message.content
            
            if "```json" in raw_content:
                json_start = raw_content.find("{")
                json_end = raw_content.rfind("}") + 1
                json_content = raw_content[json_start:json_end]
            else:
                json_content = raw_content
            
            response_json = json.loads(json_content)
            content = response_json["content"]
            urls = response_json.get("urls", [])
            
            for url_dict in urls:
                for num, link in url_dict.items():
                    content = content.replace(f"[{num}]", f'<a href="{link}">[{num}]</a>')
            
            return content
        except (json.JSONDecodeError, KeyError):
            return answer.choices[0].message.content