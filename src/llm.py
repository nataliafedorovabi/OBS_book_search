import google.generativeai as genai
from typing import List, Dict
from src.config import GEMINI_API_KEY, GEMINI_MODEL
from src.prompts import SYSTEM_PROMPT, NO_CONTEXT_RESPONSE, LOW_RELEVANCE_RESPONSE


class LLMClient:
    """Клиент для работы с Gemini LLM."""

    def __init__(self):
        genai.configure(api_key=GEMINI_API_KEY)
        self.model = genai.GenerativeModel(
            GEMINI_MODEL,
            generation_config={
                "temperature": 0.3,  # Низкая температура = меньше галлюцинаций
                "max_output_tokens": 1000,
            }
        )

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
            book = chunk["metadata"].get("book", "Неизвестная книга")
            chapter = chunk["metadata"].get("chapter", "")
            text = chunk["text"]
            context_parts.append(f"[Фрагмент {i} | {book}, {chapter}]\n{text}")

        context = "\n\n---\n\n".join(context_parts)

        prompt = f"""{SYSTEM_PROMPT}

КОНТЕКСТ ИЗ КНИГ КУРСА:
{context}

ВОПРОС СТУДЕНТА: {question}

ОТВЕТ:"""

        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"Произошла ошибка при обработке запроса. Попробуйте позже."
