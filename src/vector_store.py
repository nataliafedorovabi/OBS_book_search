import json
import logging
import re
import chromadb
from chromadb.utils import embedding_functions
from typing import List, Dict, Set
from src.config import CHROMA_DIR, PARSED_DIR, TOP_K_RESULTS, MIN_RELEVANCE_SCORE

logger = logging.getLogger(__name__)


class VectorStore:
    """Векторное хранилище с гибридным поиском (семантика + keyword)."""

    def __init__(self):
        self.client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        self.chunks_by_id: Dict[str, Dict] = {}  # Для keyword поиска

        # Многоязычная модель для русского текста
        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )

        # Загружаем чанки для keyword поиска
        self._load_chunks_for_keyword_search()

        # Пробуем получить существующую коллекцию
        try:
            self.collection = self.client.get_collection(
                name="books",
                embedding_function=self.embedding_fn
            )
            count = self.collection.count()
            logger.info(f"Загружена существующая база: {count} документов")
        except Exception as e:
            logger.info(f"Создаю новую базу из JSON...")
            self._create_from_json()

    def _load_chunks_for_keyword_search(self):
        """Загружает чанки в память для keyword поиска."""
        json_path = PARSED_DIR / "all_chunks.json"
        if json_path.exists():
            with open(json_path, 'r', encoding='utf-8') as f:
                chunks = json.load(f)
            for c in chunks:
                self.chunks_by_id[c['id']] = c
            logger.info(f"Загружено {len(self.chunks_by_id)} чанков для keyword поиска")

    def _create_from_json(self):
        """Создаёт базу из JSON файла."""
        json_path = PARSED_DIR / "all_chunks.json"

        if not json_path.exists():
            raise FileNotFoundError(f"JSON с чанками не найден: {json_path}")

        try:
            self.client.delete_collection("books")
        except:
            pass

        self.collection = self.client.create_collection(
            name="books",
            embedding_function=self.embedding_fn,
            metadata={"hnsw:space": "cosine"}
        )

        with open(json_path, 'r', encoding='utf-8') as f:
            chunks = json.load(f)

        logger.info(f"Индексация {len(chunks)} чанков...")

        batch_size = 100
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            self.collection.add(
                ids=[c["id"] for c in batch],
                documents=[c["text"] for c in batch],
                metadatas=[c["metadata"] for c in batch]
            )
            if (i + batch_size) % 500 == 0:
                logger.info(f"  Проиндексировано: {min(i + batch_size, len(chunks))}/{len(chunks)}")

        logger.info(f"Индексация завершена: {self.collection.count()} документов")

    def _extract_keywords(self, query: str) -> List[str]:
        """Извлекает ключевые слова из запроса (>3 букв, не стоп-слова)."""
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

        # Извлекаем слова длиной > 3 символов
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
            # Считаем сколько ключевых слов найдено
            matches = sum(1 for kw in keywords if kw in text_lower)
            if matches > 0:
                found.append({
                    'id': chunk_id,
                    'text': chunk['text'],
                    'metadata': chunk['metadata'],
                    'keyword_matches': matches,
                    'score': 0.5 + (matches / len(keywords)) * 0.3  # Базовый score для keyword
                })

        # Сортируем по количеству совпадений
        found.sort(key=lambda x: x['keyword_matches'], reverse=True)
        return found[:n_results]

    def search(self, query: str, n_results: int = TOP_K_RESULTS) -> List[Dict]:
        """
        Гибридный поиск: семантика + keyword.
        """
        if self.collection.count() == 0:
            return []

        # 1. Семантический поиск
        semantic_results = self.collection.query(
            query_texts=[query],
            n_results=n_results * 2  # Берём больше для объединения
        )

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
                # Найден обоими методами — повышаем score
                item = semantic_found[chunk_id]
                item['score'] = min(1.0, item['score'] + 0.2)
                item['source'] = "both"
                combined.append(item)
            elif chunk_id in semantic_found:
                combined.append(semantic_found[chunk_id])
            else:
                combined.append(keyword_found[chunk_id])

        # Сортируем по score
        combined.sort(key=lambda x: x['score'], reverse=True)

        return combined[:n_results]

    def get_count(self) -> int:
        """Возвращает количество документов в базе."""
        return self.collection.count()
