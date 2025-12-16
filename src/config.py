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
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# Настройки чанков
CHUNK_SIZE = 700  # символов в одном чанке
CHUNK_OVERLAP = 100  # перекрытие между чанками

# Настройки поиска
TOP_K_RESULTS = 5  # количество релевантных фрагментов
MIN_RELEVANCE_SCORE = 0.3  # минимальный порог релевантности (0-1)

# Модель LLM (OpenRouter)
# Бесплатные: google/gemini-flash-1.5-8b, meta-llama/llama-3.1-8b-instruct:free
# Платные: anthropic/claude-3.5-sonnet, openai/gpt-4o-mini
LLM_MODEL = os.getenv("LLM_MODEL", "google/gemini-flash-1.5-8b")

# Лимиты запросов
DAILY_REQUEST_LIMIT = 500  # запросов в день
WARNING_THRESHOLD = 0.8    # уведомить админа при 80% лимита

# Telegram ID админов (для уведомлений о лимитах)
# Можно указать несколько через запятую: 123456789,987654321
# Узнать свой ID: написать боту @userinfobot
_admin_ids = os.getenv("ADMIN_TELEGRAM_IDS", "")
ADMIN_TELEGRAM_IDS = [id.strip() for id in _admin_ids.split(",") if id.strip()]
