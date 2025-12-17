import hashlib
import logging
import requests
from typing import List, Dict
from src.config import OPENROUTER_API_KEY, LLM_MODEL, MIN_RELEVANCE_SCORE
from src.prompts import SYSTEM_PROMPT, NO_CONTEXT_RESPONSE, LOW_RELEVANCE_RESPONSE

logger = logging.getLogger(__name__)


class LLMClient:
    """Клиент для работы с LLM через OpenRouter."""

    def __init__(self):
        self.api_key = OPENROUTER_API_KEY
        self.model = LLM_MODEL
        self.api_url = "https://openrouter.ai/api/v1/chat/completions"
        # Простой кеш для экономии API вызовов (макс 100 записей)
        self._cache: Dict[str, str] = {}
        self._cache_max = 100
        logger.info(f"LLM клиент инициализирован: модель={self.model}")

    def _get_cache_key(self, question: str, chunks: List[Dict]) -> str:
        """Генерирует ключ кеша."""
        chunk_ids = "|".join(c.get("metadata", {}).get("id", str(i)) for i, c in enumerate(chunks[:3]))
        return hashlib.md5(f"{question.lower().strip()}:{chunk_ids}".encode()).hexdigest()

    def generate_answer(self, question: str, context_chunks: List[Dict]) -> str:
        """Генерирует ответ на вопрос на основе контекста из книг."""

        if not context_chunks:
            return NO_CONTEXT_RESPONSE

        # Проверяем релевантность (если все score низкие)
        if all(chunk.get('score', 0) < MIN_RELEVANCE_SCORE for chunk in context_chunks):
            return LOW_RELEVANCE_RESPONSE

        # Проверяем кеш
        cache_key = self._get_cache_key(question, context_chunks)
        if cache_key in self._cache:
            return self._cache[cache_key]

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
            logger.info(f"Отправка запроса к OpenRouter: модель={self.model}")

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
                    "max_tokens": 600,
                },
                timeout=60
            )

            logger.info(f"OpenRouter ответ: status={response.status_code}")

            if response.status_code == 200:
                data = response.json()
                answer = data["choices"][0]["message"]["content"]
                # Сохраняем в кеш (с ограничением размера)
                if len(self._cache) >= self._cache_max:
                    self._cache.pop(next(iter(self._cache)))
                self._cache[cache_key] = answer
                logger.info(f"Ответ получен успешно, длина={len(answer)}")
                return answer
            else:
                # Подробное логирование ошибки
                try:
                    error_data = response.json()
                    logger.error(f"OpenRouter ошибка: status={response.status_code}, body={error_data}")
                    error_msg = error_data.get("error", {})
                    error_text = error_msg.get("message", str(error_msg))
                    error_code = error_msg.get("code", "unknown")
                    error_type = error_msg.get("type", "unknown")
                    metadata = error_msg.get("metadata", {})

                    logger.error(f"  code={error_code}, type={error_type}")
                    logger.error(f"  metadata={metadata}")

                    return f"Ошибка API ({error_code}): {error_text}"
                except:
                    logger.error(f"OpenRouter raw ответ: {response.text[:500]}")
                    return f"Ошибка API: {response.text[:200]}"

        except requests.Timeout:
            logger.error("OpenRouter timeout (60s)")
            return "Превышено время ожидания ответа. Попробуйте позже."
        except Exception as e:
            logger.error(f"OpenRouter exception: {type(e).__name__}: {e}")
            return f"Произошла ошибка при обработке запроса. Попробуйте позже."
