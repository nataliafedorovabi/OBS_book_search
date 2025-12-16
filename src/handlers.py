from telegram import Update
from telegram.ext import ContextTypes
from src.vector_store import VectorStore
from src.llm import LLMClient


# Глобальные объекты (инициализируются в main.py)
vector_store: VectorStore = None
llm_client: LLMClient = None


def init_services(vs: VectorStore, llm: LLMClient):
    """Инициализация сервисов."""
    global vector_store, llm_client
    vector_store = vs
    llm_client = llm


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

    # Показываем статус "печатает"
    await context.bot.send_chat_action(
        chat_id=update.effective_chat.id,
        action="typing"
    )

    # Ищем релевантные фрагменты
    relevant_chunks = vector_store.search(question)

    # Генерируем ответ через LLM
    answer = llm_client.generate_answer(question, relevant_chunks)

    await update.message.reply_text(answer)
