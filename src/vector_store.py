import chromadb
from chromadb.utils import embedding_functions
from typing import List, Dict
from src.config import CHROMA_DIR, TOP_K_RESULTS, MIN_RELEVANCE_SCORE


class VectorStore:
    """Векторное хранилище для поиска по книгам."""

    def __init__(self):
        self.client = chromadb.PersistentClient(path=str(CHROMA_DIR))

        # Многоязычная модель для русского текста
        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )

        self.collection = self.client.get_or_create_collection(
            name="books",
            embedding_function=self.embedding_fn,
            metadata={"hnsw:space": "cosine"}
        )

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
