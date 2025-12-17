"""
Скрипт для локальной индексации книг через Voyage AI.
Запустить один раз на своём компьютере, потом загрузить базу на сервер.

Использование:
1. pip install voyageai chromadb
2. set VOYAGE_API_KEY=твой_ключ
3. python index_locally.py
4. Загрузить папку data/chroma_db/ на сервер
"""

import json
import os
import sys
from pathlib import Path

# Проверяем API ключ
VOYAGE_API_KEY = os.getenv("VOYAGE_API_KEY")
if not VOYAGE_API_KEY:
    print("Ошибка: установи переменную VOYAGE_API_KEY")
    print("Windows: set VOYAGE_API_KEY=твой_ключ")
    print("Linux/Mac: export VOYAGE_API_KEY=твой_ключ")
    sys.exit(1)

import voyageai
import chromadb
from chromadb.api.types import EmbeddingFunction, Documents, Embeddings

# Пути
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
PARSED_DIR = DATA_DIR / "parsed"
CHROMA_DIR = DATA_DIR / "chroma_db"


class VoyageEmbeddingFunction(EmbeddingFunction):
    """Embedding функция для Voyage AI."""

    def __init__(self, api_key: str, model: str = "voyage-multilingual-2"):
        self.client = voyageai.Client(api_key=api_key)
        self.model = model
        self.total_tokens = 0

    def __call__(self, input: Documents) -> Embeddings:
        if not input:
            return []

        result = self.client.embed(
            texts=list(input),
            model=self.model,
            input_type="document"
        )

        tokens = getattr(result, 'total_tokens', 0)
        self.total_tokens += tokens
        print(f"  Voyage API: {len(input)} текстов, {tokens} токенов (всего: {self.total_tokens})")

        return result.embeddings


def main():
    print("=" * 50)
    print("Локальная индексация книг через Voyage AI")
    print("=" * 50)

    # Проверяем JSON
    json_path = PARSED_DIR / "all_chunks.json"
    if not json_path.exists():
        print(f"Ошибка: не найден {json_path}")
        sys.exit(1)

    # Загружаем чанки
    print(f"\nЗагрузка чанков из {json_path}...")
    with open(json_path, 'r', encoding='utf-8') as f:
        chunks = json.load(f)
    print(f"Загружено: {len(chunks)} чанков")

    # Создаём ChromaDB
    print(f"\nСоздание базы в {CHROMA_DIR}...")
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    # Удаляем старые коллекции
    for name in ["books", "books_voyage"]:
        try:
            client.delete_collection(name)
            print(f"  Удалена старая коллекция: {name}")
        except:
            pass

    # Создаём embedding функцию
    embedding_fn = VoyageEmbeddingFunction(
        api_key=VOYAGE_API_KEY,
        model="voyage-multilingual-2"
    )

    # Создаём коллекцию
    collection = client.create_collection(
        name="books_voyage",
        embedding_function=embedding_fn,
        metadata={"hnsw:space": "cosine"}
    )

    # Индексируем батчами
    print(f"\nИндексация {len(chunks)} чанков...")
    batch_size = 20  # Маленькие батчи чтобы не превысить лимит

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]

        print(f"\nБатч {i//batch_size + 1}/{(len(chunks)-1)//batch_size + 1}: "
              f"чанки {i+1}-{min(i+batch_size, len(chunks))}")

        collection.add(
            ids=[c["id"] for c in batch],
            documents=[c["text"] for c in batch],
            metadatas=[c["metadata"] for c in batch]
        )

    # Готово
    print("\n" + "=" * 50)
    print(f"Индексация завершена!")
    print(f"  Документов в базе: {collection.count()}")
    print(f"  Всего токенов Voyage: {embedding_fn.total_tokens}")
    print(f"  База сохранена в: {CHROMA_DIR}")
    print("=" * 50)
    print("\nТеперь:")
    print("1. Убери data/chroma_db/ из .gitignore")
    print("2. git add data/chroma_db/")
    print("3. git commit -m 'Add pre-indexed database'")
    print("4. git push")


if __name__ == "__main__":
    main()
