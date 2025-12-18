import logging
from typing import Dict, List, Any
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ContextTypes, Application
from src.tree_search import TreeSearcher
from src.llm import LLMClient
from src.rate_limiter import RateLimiter
from src.config import ADMIN_TELEGRAM_IDS

logger = logging.getLogger(__name__)

# Глобальные объекты
searcher: TreeSearcher = None
llm_client: LLMClient = None
rate_limiter: RateLimiter = None
_bot_app: Application = None

# Хранение контекста поиска для кнопок
_search_context: Dict[int, Dict[str, Any]] = {}


def init_services(tree_searcher: TreeSearcher, llm: LLMClient, app: Application = None):
    """Инициализация сервисов."""
    global searcher, llm_client, rate_limiter, _bot_app
    searcher = tree_searcher
    llm_client = llm
    rate_limiter = RateLimiter()
    _bot_app = app
    logger.info("Сервисы инициализированы")


async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик команды /start."""
    stats = searcher.tree.get_stats() if searcher else {}

    await update.message.reply_text(
        f"Привет! Я бот-ассистент курса 'Управление организацией и персоналом'.\n\n"
        f"В базе: {stats.get('chapters', 0)} глав, {stats.get('chunks', 0)} фрагментов.\n\n"
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
    stats = searcher.tree.get_stats() if searcher else {}

    if stats:
        await update.message.reply_text(
            f"База знаний активна.\n\n"
            f"Книг: {stats.get('books', 0)}\n"
            f"Глав: {stats.get('chapters', 0)}\n"
            f"Секций: {stats.get('sections', 0)}\n"
            f"Фрагментов: {stats.get('chunks', 0)}\n\n"
            f"Версия дерева: {stats.get('version', 'N/A')}\n"
            f"Создано: {stats.get('created_at', 'N/A')[:10]}"
        )
    else:
        await update.message.reply_text("База знаний недоступна.")


async def usage_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Статистика использования (для админов)."""
    user_id = str(update.effective_user.id)

    if ADMIN_TELEGRAM_IDS and user_id not in ADMIN_TELEGRAM_IDS:
        return

    usage = rate_limiter.get_usage_info()

    await update.message.reply_text(
        f"📊 Статистика за {usage['date']}:\n\n"
        f"Запросов: {usage['requests_today']}/{usage['limit']}\n"
        f"Использовано: {usage['percent_used']}%\n"
        f"Осталось: {usage['remaining']}"
    )


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик вопросов пользователя."""
    question = update.message.text.strip()

    if not question:
        return

    if not searcher:
        await update.message.reply_text("База знаний временно недоступна.")
        return

    # Проверяем лимит
    if not rate_limiter.can_make_request():
        await update.message.reply_text(
            "Достигнут дневной лимит запросов. Попробуйте завтра."
        )
        return

    # Показываем "печатает"
    await context.bot.send_chat_action(
        chat_id=update.effective_chat.id,
        action="typing"
    )

    logger.info(f"Вопрос: {question[:50]}...")

    # Поиск по дереву
    results = searcher.search(question, top_chapters=4, top_chunks=6)

    if not results:
        await update.message.reply_text(
            "В материалах курса нет информации по этому вопросу. "
            "Попробуйте переформулировать вопрос."
        )
        return

    # Формируем контекст для LLM
    context_chunks = []
    chapters_found = {}

    for r in results:
        context_chunks.append({
            'text': r.text,
            'metadata': {
                'book_title': r.book_title,
                'chapter': r.chapter_title,
                'section': r.section_title
            },
            'score': r.score
        })

        # Группируем по главам
        ch_title = r.chapter_title
        if ch_title not in chapters_found:
            chapters_found[ch_title] = {
                'summary': r.chapter_summary,
                'chunks': []
            }
        chapters_found[ch_title]['chunks'].append(r)

    logger.info(f"Найдено: {len(results)} чанков из {len(chapters_found)} глав")

    # Генерируем ответ
    answer = llm_client.generate_answer(question, context_chunks, is_expanded_search=True)
    rate_limiter.record_request()

    await update.message.reply_text(answer, parse_mode="Markdown")


async def button_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик нажатий на кнопки."""
    query = update.callback_query
    await query.answer()

    user_id = update.effective_user.id
    search_ctx = _search_context.get(user_id)

    if not search_ctx:
        await query.edit_message_reply_markup(reply_markup=None)
        await query.message.reply_text("Контекст устарел. Задайте вопрос заново.")
        return

    callback_data = query.data

    if callback_data.startswith("chapter_"):
        # Подробнее по главе
        chapter_idx = int(callback_data.split("_")[1])
        chapters_list = list(search_ctx['chapters'].keys())

        if chapter_idx >= len(chapters_list):
            await query.message.reply_text("Глава не найдена.")
            return

        chapter_name = chapters_list[chapter_idx]
        chapter_data = search_ctx['chapters'][chapter_name]

        # Показываем summary главы и найденные фрагменты
        summary = chapter_data.get('summary', '')
        chunks = chapter_data.get('chunks', [])

        response = f"📖 *{chapter_name}*\n\n"
        if summary:
            response += f"_{summary[:300]}..._\n\n" if len(summary) > 300 else f"_{summary}_\n\n"

        response += "**Найденные фрагменты:**\n\n"
        for i, chunk in enumerate(chunks[:3], 1):
            text_preview = chunk.text[:200] + "..." if len(chunk.text) > 200 else chunk.text
            response += f"{i}. {text_preview}\n\n"

        await query.message.reply_text(response, parse_mode="Markdown")

    elif callback_data == "search_more":
        # Расширенный поиск
        original_query = search_ctx['query']

        await context.bot.send_chat_action(
            chat_id=update.effective_chat.id,
            action="typing"
        )

        # Ищем в других главах
        more_results = searcher.search(original_query, top_chapters=6, top_chunks=8)

        # Исключаем уже показанные
        shown_ids = {r.chunk_id for r in search_ctx['results']}
        new_results = [r for r in more_results if r.chunk_id not in shown_ids]

        if not new_results:
            await query.message.reply_text(
                "Других материалов не найдено. Попробуйте переформулировать вопрос."
            )
            return

        # Формируем контекст
        context_chunks = [{
            'text': r.text,
            'metadata': {
                'book_title': r.book_title,
                'chapter': r.chapter_title
            },
            'score': r.score
        } for r in new_results[:5]]

        answer = llm_client.generate_answer(original_query, context_chunks, is_expanded_search=True)
        rate_limiter.record_request()

        await query.message.reply_text(f"🔍 Дополнительные результаты:\n\n{answer}")
