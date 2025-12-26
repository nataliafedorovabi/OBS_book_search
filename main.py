import logging
import re
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
    handle_callback
)


class SecretFilter(logging.Filter):
    """Фильтр для маскирования секретов в логах."""

    PATTERNS = [
        (re.compile(r'bot\d+:[A-Za-z0-9_-]+'), 'bot***:***'),
        (re.compile(r'Bearer\s+[A-Za-z0-9_-]+'), 'Bearer ***'),
        (re.compile(r'sk-[A-Za-z0-9_-]+'), 'sk-***'),
        (re.compile(r'api[_-]?key["\s:=]+[A-Za-z0-9_-]+', re.I), 'api_key=***'),
    ]

    def filter(self, record):
        if record.msg:
            msg = str(record.msg)
            for pattern, replacement in self.PATTERNS:
                msg = pattern.sub(replacement, msg)
            record.msg = msg
        if record.args:
            new_args = []
            for arg in record.args:
                if isinstance(arg, str):
                    for pattern, replacement in self.PATTERNS:
                        arg = pattern.sub(replacement, arg)
                new_args.append(arg)
            record.args = tuple(new_args)
        return True


logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

root_logger = logging.getLogger()
for handler in root_logger.handlers:
    handler.addFilter(SecretFilter())

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("requests").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)


def main():
    if not TELEGRAM_TOKEN:
        logger.error("TELEGRAM_TOKEN не задан!")
        return

    logger.info("Загрузка дерева контекста...")
    tree = ContextTree()
    if not tree.load():
        logger.error("Не удалось загрузить дерево!")
        return

    stats = tree.get_stats()
    logger.info(f"Дерево загружено: {stats['chapters']} глав, {stats['chunks']} чанков")

    logger.info("Инициализация LLM клиента...")
    llm_client = LLMClient()

    searcher = TreeSearcher(tree, llm_client)

    application = Application.builder().token(TELEGRAM_TOKEN).build()

    init_services(searcher, llm_client, application)

    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("status", status_command))
    application.add_handler(CommandHandler("usage", usage_command))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    application.add_handler(CallbackQueryHandler(handle_callback))

    logger.info("Бот запущен!")
    application.run_polling(allowed_updates=["message", "callback_query"])


if __name__ == "__main__":
    main()
