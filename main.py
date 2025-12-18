import logging
from telegram.ext import Application, CommandHandler, MessageHandler, CallbackQueryHandler, filters

from src.config import TELEGRAM_TOKEN
from src.tree_search import ContextTree, TreeSearcher
from src.llm import LLMClient
from src.handlers import (
    init_services,
    start_command,
    help_command,
    status_command,
    usage_command,
    handle_message,
    button_callback
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

    # Загружаем дерево контекста
    logger.info("Загрузка дерева контекста...")
    tree = ContextTree()
    if not tree.load():
        logger.error("Не удалось загрузить дерево! Проверьте bot/data/context_tree.json")
        return

    stats = tree.get_stats()
    logger.info(f"Дерево загружено: {stats['chapters']} глав, {stats['chunks']} чанков")

    # Инициализируем LLM
    logger.info("Инициализация LLM клиента...")
    llm_client = LLMClient()

    # Создаём поисковик
    searcher = TreeSearcher(tree, llm_client)

    # Создаём приложение бота
    application = Application.builder().token(TELEGRAM_TOKEN).build()

    # Передаём сервисы в обработчики
    init_services(searcher, llm_client, application)

    # Регистрируем обработчики
    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("status", status_command))
    application.add_handler(CommandHandler("usage", usage_command))
    application.add_handler(CallbackQueryHandler(button_callback))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    # Запускаем бота
    logger.info("Бот запущен!")
    application.run_polling(allowed_updates=["message", "callback_query"])


if __name__ == "__main__":
    main()
