import logging
from telegram.ext import Application, CommandHandler, MessageHandler, filters

from src.config import TELEGRAM_TOKEN, CHROMA_DIR
from src.vector_store import VectorStore
from src.llm import LLMClient
from src.handlers import (
    init_services,
    start_command,
    help_command,
    status_command,
    usage_command,
    handle_message
)

# Настройка логирования
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)


def main():
    """Запуск бота."""

    # Проверяем токены
    if not TELEGRAM_TOKEN:
        logger.error("TELEGRAM_TOKEN не задан!")
        return

    # Проверяем наличие базы данных
    if not CHROMA_DIR.exists():
        logger.error(f"База данных не найдена: {CHROMA_DIR}")
        logger.error("Сначала запустите локальный скрипт парсинга книг!")
        return

    # Инициализируем сервисы
    logger.info("Инициализация векторной базы...")
    vector_store = VectorStore()

    docs_count = vector_store.get_count()
    if docs_count == 0:
        logger.error("База знаний пуста! Сначала запустите парсинг книг.")
        return

    logger.info(f"Загружено {docs_count} фрагментов из книг.")

    logger.info("Инициализация LLM клиента...")
    llm_client = LLMClient()

    # Передаём сервисы в обработчики
    init_services(vector_store, llm_client)

    # Создаём приложение бота
    application = Application.builder().token(TELEGRAM_TOKEN).build()

    # Регистрируем обработчики
    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("status", status_command))
    application.add_handler(CommandHandler("usage", usage_command))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    # Запускаем бота
    logger.info("Бот запущен!")
    application.run_polling(allowed_updates=["message"])


if __name__ == "__main__":
    main()
