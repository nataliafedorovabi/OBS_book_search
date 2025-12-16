import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Пути
BASE_DIR = Path(__file__).parent.parent  # bot/
DATA_DIR = BASE_DIR / "data"
PARSED_DIR = DATA_DIR / "parsed"
CHROMA_DIR = DATA_DIR / "chroma_db"

# Токены (из переменных окружения Railway)
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Настройки чанков
CHUNK_SIZE = 700  # символов в одном чанке
CHUNK_OVERLAP = 100  # перекрытие между чанками

# Настройки поиска
TOP_K_RESULTS = 5  # количество релевантных фрагментов
MIN_RELEVANCE_SCORE = 0.3  # минимальный порог релевантности (0-1)

# Модель Gemini
GEMINI_MODEL = "gemini-1.5-flash"
