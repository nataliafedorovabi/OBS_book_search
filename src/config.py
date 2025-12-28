import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Пути
BASE_DIR = Path(__file__).parent.parent  # bot/
DATA_DIR = BASE_DIR / "data"

# Путь для статистики (для Railway Volume: /data)
STATS_DIR = Path(os.getenv("STATS_DIR", str(DATA_DIR)))

# Токены (из переменных окружения Railway)
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# Настройки поиска
TOP_K_RESULTS = 5  # количество релевантных фрагментов
MIN_RELEVANCE_SCORE = 0.15  # минимальный порог релевантности

# Модель LLM (OpenRouter)
# gpt-4o-mini: ~$0.00015/1K токенов, стабильная, быстрая
LLM_MODEL = os.getenv("LLM_MODEL", "openai/gpt-4o-mini")

# Лимиты запросов
DAILY_REQUEST_LIMIT = 500  # запросов в день
WARNING_THRESHOLD = 0.8    # уведомить админа при 80% лимита

# Telegram ID админов (для уведомлений о лимитах)
# Можно указать несколько через запятую: 123456789,987654321
# Узнать свой ID: написать боту @userinfobot
_admin_ids = os.getenv("ADMIN_TELEGRAM_ID", "")
ADMIN_TELEGRAM_ID = [id.strip() for id in _admin_ids.split(",") if id.strip()]
