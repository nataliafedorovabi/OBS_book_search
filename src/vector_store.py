import json
import logging
import voyageai
import chromadb
from chromadb.api.types import EmbeddingFunction, Documents, Embeddings
from typing import List, Dict, Set, Optional, Callable
from src.config import CHROMA_DIR, PARSED_DIR, TOP_K_RESULTS, MIN_RELEVANCE_SCORE, ENABLE_HYBRID_SEARCH, VOYAGE_API_KEY

logger = logging.getLogger(__name__)

# Глобальный callback для уведомлений (устанавливается из handlers.py)
_admin_notify_callback: Optional[Callable[[str], None]] = None


def set_admin_notify_callback(callback: Callable[[str], None]):
    """Устанавливает callback для уведомления админа."""
    global _admin_notify_callback
    _admin_notify_callback = callback


class VoyageEmbeddingFunction(EmbeddingFunction):
    """Кастомная embedding функция для Voyage AI с логированием."""

    def __init__(self, api_key: str, model: str = "voyage-multilingual-2"):
        self.client = voyageai.Client(api_key=api_key)
        self.model = model
        self.total_tokens_used = 0
        self.request_count = 0
        logger.info(f"Voyage AI инициализирован: модель={model}")

    def __call__(self, input: Documents) -> Embeddings:
        """Создаёт эмбеддинги для списка текстов."""
        if not input:
            return []

        try:
            self.request_count += 1
            logger.info(f"Voyage API запрос #{self.request_count}: {len(input)} текстов")

            result = self.client.embed(
                texts=list(input),
                model=self.model,
                input_type="document"
            )

            # Логируем использование токенов
            tokens_used = getattr(result, 'total_tokens', 0)
            self.total_tokens_used += tokens_used
            logger.info(f"Voyage API ответ: {len(result.embeddings)} эмбеддингов, "
                       f"токенов={tokens_used}, всего={self.total_tokens_used}")

            return result.embeddings

        except voyageai.error.RateLimitError as e:
            error_msg = f"Voyage AI лимит превышен: {e}"
            logger.error(error_msg)
            self._notify_admin(f"ВНИМАНИЕ! {error_msg}")
            raise

        except voyageai.error.InvalidRequestError as e:
            error_msg = f"Voyage AI ошибка запроса: {e}"
            logger.error(error_msg)
            raise

        except Exception as e:
            error_msg = f"Voyage AI неизвестная ошибка: {type(e).__name__}: {e}"
            logger.error(error_msg)
            self._notify_admin(f"Ошибка Voyage AI: {error_msg}")
            raise

    def _notify_admin(self, message: str):
        """Уведомляет админа о проблеме."""
        if _admin_notify_callback:
            try:
                _admin_notify_callback(message)
            except Exception as e:
                logger.error(f"Не удалось уведомить админа: {e}")

    def get_stats(self) -> dict:
        """Возвращает статистику использования."""
        return {
            "total_tokens": self.total_tokens_used,
            "request_count": self.request_count,
            "model": self.model
        }


class VectorStore:
    """Векторное хранилище с гибридным поиском (семантика + keyword)."""

    def __init__(self):
        self.client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        self.chunks_by_id: Dict[str, Dict] = {}

        # Voyage AI для эмбеддингов (вместо локальной модели - экономим 800MB RAM)
        if not VOYAGE_API_KEY:
            raise ValueError("VOYAGE_API_KEY не задан! Добавьте в переменные окружения.")

        self.embedding_fn = VoyageEmbeddingFunction(
            api_key=VOYAGE_API_KEY,
            model="voyage-multilingual-2"  # Лучшая модель для русского
        )

        # Пробуем получить существующую коллекцию
        try:
            self.collection = self.client.get_collection(
                name="books_voyage",  # Новое имя - эмбеддинги несовместимы со старыми
                embedding_function=self.embedding_fn
            )
            count = self.collection.count()
            logger.info(f"Загружена существующая база: {count} документов")
        except Exception as e:
            logger.info(f"Создаю новую базу из JSON...")
            self._create_from_json()

        # Загружаем данные для keyword поиска
        if ENABLE_HYBRID_SEARCH:
            self._load_chunks_for_keyword_search()
        else:
            logger.info("Гибридный поиск отключён (ENABLE_HYBRID_SEARCH=false)")

    def _load_chunks_for_keyword_search(self):
        """Загружает чанки из ChromaDB для keyword поиска."""
        try:
            all_data = self.collection.get(include=["documents", "metadatas"])
            for i, chunk_id in enumerate(all_data['ids']):
                self.chunks_by_id[chunk_id] = {
                    'id': chunk_id,
                    'text': all_data['documents'][i],
                    'metadata': all_data['metadatas'][i]
                }
            logger.info(f"Загружено {len(self.chunks_by_id)} чанков для keyword поиска")
        except Exception as e:
            logger.warning(f"Не удалось загрузить чанки для keyword поиска: {e}")

    def _create_from_json(self):
        """Создаёт базу из JSON файла."""
        json_path = PARSED_DIR / "all_chunks.json"

        if not json_path.exists():
            raise FileNotFoundError(f"JSON с чанками не найден: {json_path}")

        # Удаляем старые коллекции
        for name in ["books", "books_voyage"]:
            try:
                self.client.delete_collection(name)
            except:
                pass

        self.collection = self.client.create_collection(
            name="books_voyage",
            embedding_function=self.embedding_fn,
            metadata={"hnsw:space": "cosine"}
        )

        with open(json_path, 'r', encoding='utf-8') as f:
            chunks = json.load(f)

        logger.info(f"Индексация {len(chunks)} чанков через Voyage AI...")

        batch_size = 50  # Voyage рекомендует меньшие батчи
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            self.collection.add(
                ids=[c["id"] for c in batch],
                documents=[c["text"] for c in batch],
                metadatas=[c["metadata"] for c in batch]
            )
            logger.info(f"  Проиндексировано: {min(i + batch_size, len(chunks))}/{len(chunks)}")

        logger.info(f"Индексация завершена: {self.collection.count()} документов")
        logger.info(f"Voyage AI статистика: {self.embedding_fn.get_stats()}")

    def _extract_keywords(self, query: str) -> List[str]:
        """Извлекает ключевые слова из запроса."""
        import re
        stop_words = {
            'что', 'как', 'где', 'когда', 'какой', 'какая', 'какие', 'каких',
            'это', 'эта', 'эти', 'этот', 'там', 'тут', 'здесь', 'была', 'было',
            'были', 'быть', 'есть', 'нет', 'все', 'всё', 'весь', 'вся', 'они',
            'она', 'оно', 'его', 'его', 'для', 'при', 'про', 'над', 'под',
            'между', 'через', 'после', 'перед', 'можно', 'нужно', 'надо',
            'помню', 'скажи', 'найди', 'покажи', 'расскажи', 'объясни',
            'искать', 'книга', 'книге', 'глава', 'главе', 'схема', 'схемы',
            'таблица', 'таблицы', 'рисунок', 'модель', 'модели'
        }
        words = re.findall(r'[а-яёa-z]+', query.lower())
        keywords = [w for w in words if len(w) > 3 and w not in stop_words]
        return keywords

    def _keyword_search(self, query: str, n_results: int = 10) -> List[Dict]:
        """Поиск по ключевым словам."""
        keywords = self._extract_keywords(query)
        if not keywords:
            return []

        found = []
        for chunk_id, chunk in self.chunks_by_id.items():
            text_lower = chunk['text'].lower()
            matches = sum(1 for kw in keywords if kw in text_lower)
            if matches > 0:
                found.append({
                    'id': chunk_id,
                    'text': chunk['text'],
                    'metadata': chunk['metadata'],
                    'keyword_matches': matches,
                    'score': 0.5 + (matches / len(keywords)) * 0.3
                })

        found.sort(key=lambda x: x['keyword_matches'], reverse=True)
        return found[:n_results]

    def search(self, query: str, n_results: int = TOP_K_RESULTS) -> List[Dict]:
        """Поиск: семантика + keyword (если включён гибридный режим)."""
        if self.collection.count() == 0:
            return []

        # 1. Семантический поиск через Voyage AI
        search_count = n_results * 2 if ENABLE_HYBRID_SEARCH else n_results

        try:
            semantic_results = self.collection.query(
                query_texts=[query],
                n_results=search_count
            )
        except Exception as e:
            logger.error(f"Ошибка семантического поиска: {e}")
            # При ошибке Voyage пробуем только keyword поиск
            if ENABLE_HYBRID_SEARCH:
                return self._keyword_search(query, n_results)
            return []

        semantic_found = {}
        if semantic_results and semantic_results['documents']:
            for i, doc in enumerate(semantic_results['documents'][0]):
                distance = semantic_results['distances'][0][i] if semantic_results.get('distances') else 0
                score = 1 - distance
                chunk_id = semantic_results['ids'][0][i]

                if score >= MIN_RELEVANCE_SCORE:
                    semantic_found[chunk_id] = {
                        "text": doc,
                        "metadata": semantic_results['metadatas'][0][i],
                        "score": score,
                        "source": "semantic"
                    }

        # Если гибридный поиск выключен
        if not ENABLE_HYBRID_SEARCH:
            results = list(semantic_found.values())
            results.sort(key=lambda x: x['score'], reverse=True)
            return results[:n_results]

        # 2. Keyword поиск
        keyword_results = self._keyword_search(query, n_results * 2)

        keyword_found = {}
        for item in keyword_results:
            chunk_id = item['id']
            if chunk_id not in keyword_found:
                keyword_found[chunk_id] = {
                    "text": item['text'],
                    "metadata": item['metadata'],
                    "score": item['score'],
                    "source": "keyword"
                }

        # 3. Объединяем результаты
        all_ids: Set[str] = set(semantic_found.keys()) | set(keyword_found.keys())

        combined = []
        for chunk_id in all_ids:
            if chunk_id in semantic_found and chunk_id in keyword_found:
                item = semantic_found[chunk_id]
                item['score'] = min(1.0, item['score'] + 0.2)
                item['source'] = "both"
                combined.append(item)
            elif chunk_id in semantic_found:
                combined.append(semantic_found[chunk_id])
            else:
                combined.append(keyword_found[chunk_id])

        combined.sort(key=lambda x: x['score'], reverse=True)
        return combined[:n_results]

    def get_count(self) -> int:
        """Возвращает количество документов в базе."""
        return self.collection.count()

    def get_embedding_stats(self) -> dict:
        """Возвращает статистику использования Voyage AI."""
        return self.embedding_fn.get_stats()
