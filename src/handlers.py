from telegram import Update
from telegram.ext import ContextTypes
from src.vector_store import VectorStore
from src.llm import LLMClient
from src.rate_limiter import RateLimiter
from src.config import ADMIN_TELEGRAM_IDS


# Глобальные объекты (инициализируются в main.py)
vector_store: VectorStore = None
llm_client: LLMClient = None
rate_limiter: RateLimiter = None


def init_services(vs: VectorStore, llm: LLMClient):
    """Инициализация сервисов."""
    global vector_store, llm_client, rate_limiter
    vector_store = vs
    llm_client = llm
    rate_limiter = RateLimiter()


async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик команды /start."""
    docs_count = vector_store.get_count() if vector_store else 0

    await update.message.reply_text(
        f"Привет! Я бот-ассистент курса 'Управление организацией и персоналом'.\n\n"
        f"В базе: {docs_count} фрагментов из книг курса.\n\n"
        f"Задайте вопрос по материалам курса, и я найду ответ в книгах.\n\n"
        f"Команды:\n"
        f"/help - как задавать вопросы\n"
        f"/status - статус базы знаний"
    )


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик команды /help."""
    await update.message.reply_text(
        "Как задавать вопросы:\n\n"
        "- Формулируйте конкретно: 'Какие роли выполняет менеджер?'\n"
        "- Можно спрашивать про термины: 'Что такое делегирование?'\n"
        "- Можно просить сравнить: 'В чём разница между лидером и менеджером?'\n\n"
        "Я отвечаю только на основе книг курса. "
        "Если информации нет в материалах, я честно об этом скажу."
    )


async def status_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик команды /status."""
    docs_count = vector_store.get_count() if vector_store else 0

    if docs_count > 0:
        await update.message.reply_text(
            f"База знаний активна.\n"
            f"Загружено: {docs_count} фрагментов из книг курса."
        )
    else:
        await update.message.reply_text(
            "База знаний пуста. Обратитесь к администратору."
        )


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик текстовых сообщений - вопросов пользователя."""
    question = update.message.text.strip()

    if not question:
        return

    if not vector_store or vector_store.get_count() == 0:
        await update.message.reply_text(
            "База знаний временно недоступна. Попробуйте позже."
        )
        return

    # Проверяем лимит запросов
    if not rate_limiter.can_make_request():
        await update.message.reply_text(
            "Достигнут дневной лимит запросов. Попробуйте завтра.\n"
            "Приносим извинения за неудобства."
        )
        return

    # Показываем статус "печатает"
    await context.bot.send_chat_action(
        chat_id=update.effective_chat.id,
        action="typing"
    )

    # Ищем релевантные фрагменты
    relevant_chunks = vector_store.search(question)

    # Генерируем ответ через LLM
    answer = llm_client.generate_answer(question, relevant_chunks)

    # Записываем запрос
    rate_limiter.record_request()

    # Уведомляем админов если приближаемся к лимиту
    if rate_limiter.should_warn_admin() and ADMIN_TELEGRAM_IDS:
        usage = rate_limiter.get_usage_info()
        for admin_id in ADMIN_TELEGRAM_IDS:
            try:
                await context.bot.send_message(
                    chat_id=admin_id,
                    text=f"Внимание! Использовано {usage['percent_used']}% дневного лимита.\n"
                         f"Запросов: {usage['requests_today']}/{usage['limit']}\n"
                         f"Осталось: {usage['remaining']}"
                )
            except:
                pass
        rate_limiter.mark_warning_sent()

    await update.message.reply_text(answer)


async def usage_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик команды /usage - статистика использования (для админов)."""
    user_id = str(update.effective_user.id)

    # Команда только для админов
    if ADMIN_TELEGRAM_IDS and user_id not in ADMIN_TELEGRAM_IDS:
        return

    usage = rate_limiter.get_usage_info()
    await update.message.reply_text(
        f"Статистика за {usage['date']}:\n\n"
        f"Запросов: {usage['requests_today']}/{usage['limit']}\n"
        f"Использовано: {usage['percent_used']}%\n"
        f"Осталось: {usage['remaining']}"
    )
