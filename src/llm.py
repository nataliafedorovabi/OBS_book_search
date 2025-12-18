import hashlib
import json
import logging
import time
import requests
from typing import List, Dict, Optional
from src.config import OPENROUTER_API_KEY, LLM_MODEL, MIN_RELEVANCE_SCORE
from src.prompts import (
    SYSTEM_PROMPT, SYSTEM_PROMPT_WITH_ALTERNATIVES,
    UNDERSTAND_QUERY_PROMPT,
    NO_CONTEXT_RESPONSE, LOW_RELEVANCE_RESPONSE
)

logger = logging.getLogger(__name__)


class LLMClient:
    """Клиент для работы с LLM через OpenRouter."""

    def __init__(self):
        self.api_key = OPENROUTER_API_KEY
        self.model = LLM_MODEL
        self.api_url = "https://openrouter.ai/api/v1/chat/completions"
        self._cache: Dict[str, str] = {}
        self._cache_max = 100
        logger.info(f"LLM клиент инициализирован: модель={self.model}")

    def _call_llm(self, system_prompt: str, user_message: str,
                  max_tokens: int = 600, temperature: float = 0.3) -> Optional[str]:
        """Базовый вызов LLM."""
        max_retries = 3
        retry_delay = 2

        for attempt in range(max_retries):
            try:
                logger.info(f"Запрос к OpenRouter, попытка {attempt + 1}/{max_retries}")

                response = requests.post(
                    self.api_url,
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": self.model,
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_message}
                        ],
                        "temperature": temperature,
                        "max_tokens": max_tokens,
                    },
                    timeout=60
                )

                if response.status_code == 429:
                    if attempt < max_retries - 1:
                        wait_time = retry_delay * (2 ** attempt)
                        logger.warning(f"Rate limit (429), ожидание {wait_time}с...")
                        time.sleep(wait_time)
                        continue
                    return None

                if response.status_code == 200:
                    data = response.json()
                    return data["choices"][0]["message"]["content"]
                else:
                    logger.error(f"OpenRouter ошибка: status={response.status_code}")
                    return None

            except Exception as e:
                logger.error(f"OpenRouter exception: {type(e).__name__}: {e}")
                if attempt < max_retries - 1:
                    continue
                return None

        return None

    def understand_query(self, query: str, chapters_info: str = "") -> Dict:
        """
        Анализирует вопрос и возвращает ключевые термины для поиска.

        Args:
            query: Вопрос пользователя
            chapters_info: Информация о главах (опционально)

        Returns:
            {"search_terms": [...]}
        """
        logger.info(f"Анализ вопроса: {query}")

        prompt = UNDERSTAND_QUERY_PROMPT.format(
            chapters_info=chapters_info,
            query=query
        )

        result = self._call_llm(
            system_prompt="Ты помощник. Отвечай только JSON.",
            user_message=prompt,
            max_tokens=200,
            temperature=0.1
        )

        if not result:
            logger.warning("Не удалось проанализировать вопрос")
            return {"chapters": [], "search_terms": [query]}

        try:
            result = result.strip()
            if result.startswith("```"):
                result = result.split("```")[1]
                if result.startswith("json"):
                    result = result[4:]

            parsed = json.loads(result)
            chapters = parsed.get('chapters', [])
            terms = parsed.get('search_terms', [query])
            logger.info(f"Понял вопрос: главы={chapters}, термины={terms}")
            return {
                "chapters": chapters,
                "search_terms": terms if terms else [query]
            }
        except json.JSONDecodeError as e:
            logger.warning(f"Не удалось распарсить JSON: {e}")
            return {"chapters": [], "search_terms": [query]}

    def _get_cache_key(self, question: str, chunks: List[Dict]) -> str:
        """Генерирует ключ кеша."""
        chunk_ids = "|".join(c.get("metadata", {}).get("id", str(i)) for i, c in enumerate(chunks[:3]))
        return hashlib.md5(f"{question.lower().strip()}:{chunk_ids}".encode()).hexdigest()

    def generate_answer(self, question: str, context_chunks: List[Dict],
                       is_expanded_search: bool = False) -> str:
        """
        Генерирует ответ на вопрос на основе контекста из книг.

        Args:
            question: Вопрос пользователя
            context_chunks: Найденные фрагменты
            is_expanded_search: True если это результат расширенного поиска
        """
        if not context_chunks:
            return NO_CONTEXT_RESPONSE

        # Проверяем релевантность
        if all(chunk.get('score', 0) < MIN_RELEVANCE_SCORE for chunk in context_chunks):
            return LOW_RELEVANCE_RESPONSE

        # Проверяем кеш
        cache_key = self._get_cache_key(question, context_chunks)
        if cache_key in self._cache:
            logger.info("Ответ из кеша")
            return self._cache[cache_key]

        # Формируем контекст
        context_parts = []
        for i, chunk in enumerate(context_chunks, 1):
            book = chunk["metadata"].get("book_title", chunk["metadata"].get("book", "Книга"))
            chapter = chunk["metadata"].get("chapter", "")
            text = chunk["text"]
            context_parts.append(f"[Фрагмент {i} | {book}, {chapter}]\n{text}")

        context = "\n\n---\n\n".join(context_parts)

        # Используем разные промпты в зависимости от типа поиска
        if is_expanded_search:
            system_prompt = SYSTEM_PROMPT_WITH_ALTERNATIVES
            user_message = f"""КОНТЕКСТ ИЗ КНИГ КУРСА:
{context}

ВОПРОС СТУДЕНТА: {question}

ВАЖНО: Точный термин из вопроса может отсутствовать в материалах.
Если так — предложи связанные концепции из найденных фрагментов.

ОТВЕТ:"""
        else:
            system_prompt = SYSTEM_PROMPT
            user_message = f"""КОНТЕКСТ ИЗ КНИГ КУРСА:
{context}

ВОПРОС СТУДЕНТА: {question}

ОТВЕТ:"""

        answer = self._call_llm(system_prompt, user_message)

        if answer:
            # Сохраняем в кеш
            if len(self._cache) >= self._cache_max:
                self._cache.pop(next(iter(self._cache)))
            self._cache[cache_key] = answer
            logger.info(f"Ответ получен, длина={len(answer)}")
            return answer

        return "Не удалось получить ответ. Попробуйте позже."
