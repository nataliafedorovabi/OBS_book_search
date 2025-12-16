import json
import logging
import chromadb
from chromadb.utils import embedding_functions
from typing import List, Dict
from src.config import CHROMA_DIR, PARSED_DIR, TOP_K_RESULTS, MIN_RELEVANCE_SCORE

logger = logging.getLogger(__name__)


class VectorStore:
    """Векторное хранилище для поиска по книгам."""

    def __init__(self):
        self.client = chromadb.PersistentClient(path=str(CHROMA_DIR))

        # Многоязычная модель для русского текста
        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )

        # Пробуем получить существующую коллекцию
        try:
            self.collection = self.client.get_collection(
                name="books",
                embedding_function=self.embedding_fn
            )
            # Проверяем что база работает
            count = self.collection.count()
            logger.info(f"Загружена существующая база: {count} документов")
        except Exception as e:
            logger.info(f"Создаю новую базу из JSON...")
            self._create_from_json()

    def _create_from_json(self):
        """Создаёт базу из JSON файла."""
        json_path = PARSED_DIR / "all_chunks.json"

        if not json_path.exists():
            raise FileNotFoundError(f"JSON с чанками не найден: {json_path}")

        # Удаляем старую коллекцию если есть
        try:
            self.client.delete_collection("books")
        except:
            pass

        # Создаём новую
        self.collection = self.client.create_collection(
            name="books",
            embedding_function=self.embedding_fn,
            metadata={"hnsw:space": "cosine"}
        )

        # Загружаем чанки
        with open(json_path, 'r', encoding='utf-8') as f:
            chunks = json.load(f)

        logger.info(f"Индексация {len(chunks)} чанков...")

        # Добавляем пачками
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

    def search(self, query: str, n_results: int = TOP_K_RESULTS) -> List[Dict]:
        """
        Поиск релевантных фрагментов по запросу.
        Возвращает только фрагменты с score выше MIN_RELEVANCE_SCORE.
        """
        if self.collection.count() == 0:
            return []

        results = self.collection.query(
            query_texts=[query],
            n_results=n_results
        )

        found = []
        if results and results['documents']:
            for i, doc in enumerate(results['documents'][0]):
                distance = results['distances'][0][i] if results.get('distances') else 0
                score = 1 - distance  # Cosine distance → similarity score

                # Фильтруем нерелевантные результаты
                if score >= MIN_RELEVANCE_SCORE:
                    found.append({
                        "text": doc,
                        "metadata": results['metadatas'][0][i],
                        "score": score
                    })

        return found

    def get_count(self) -> int:
        """Возвращает количество документов в базе."""
        return self.collection.count()
