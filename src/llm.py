import requests
from typing import List, Dict
from src.config import OPENROUTER_API_KEY, LLM_MODEL
from src.prompts import SYSTEM_PROMPT, NO_CONTEXT_RESPONSE, LOW_RELEVANCE_RESPONSE


class LLMClient:
    """Клиент для работы с LLM через OpenRouter."""

    def __init__(self):
        self.api_key = OPENROUTER_API_KEY
        self.model = LLM_MODEL
        self.api_url = "https://openrouter.ai/api/v1/chat/completions"

    def generate_answer(self, question: str, context_chunks: List[Dict]) -> str:
        """Генерирует ответ на вопрос на основе контекста из книг."""

        if not context_chunks:
            return NO_CONTEXT_RESPONSE

        # Проверяем релевантность (если все score низкие)
        if all(chunk.get('score', 0) < 0.4 for chunk in context_chunks):
            return LOW_RELEVANCE_RESPONSE

        # Формируем контекст
        context_parts = []
        for i, chunk in enumerate(context_chunks, 1):
            book = chunk["metadata"].get("book_title", chunk["metadata"].get("book", "Книга"))
            chapter = chunk["metadata"].get("chapter", "")
            text = chunk["text"]
            context_parts.append(f"[Фрагмент {i} | {book}, {chapter}]\n{text}")

        context = "\n\n---\n\n".join(context_parts)

        user_message = f"""КОНТЕКСТ ИЗ КНИГ КУРСА:
{context}

ВОПРОС СТУДЕНТА: {question}

ОТВЕТ:"""

        try:
            response = requests.post(
                self.api_url,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_message}
                    ],
                    "temperature": 0.3,
                    "max_tokens": 1000,
                },
                timeout=60
            )

            if response.status_code == 200:
                data = response.json()
                return data["choices"][0]["message"]["content"]
            else:
                error = response.json().get("error", {}).get("message", response.text)
                return f"Ошибка API: {error}"

        except requests.Timeout:
            return "Превышено время ожидания ответа. Попробуйте позже."
        except Exception as e:
            return f"Произошла ошибка при обработке запроса. Попробуйте позже."
