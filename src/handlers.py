import logging
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ContextTypes, Application
from src.tree_search import TreeSearcher
from src.llm import LLMClient
from src.rate_limiter import RateLimiter
from src.config import ADMIN_TELEGRAM_IDS
from src.chapters import get_book_display_name

logger = logging.getLogger(__name__)

searcher: TreeSearcher = None
llm_client: LLMClient = None
rate_limiter: RateLimiter = None
search_results_cache = {}


def init_services(tree_searcher: TreeSearcher, llm: LLMClient, app: Application = None):
    global searcher, llm_client, rate_limiter
    searcher = tree_searcher
    llm_client = llm
    rate_limiter = RateLimiter()
    logger.info("Сервисы инициализированы")


async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    stats = searcher.tree.get_stats() if searcher else {}
    text = (
        "Привет! Я бот-ассистент курсов:\n"
        "- R628 Управление организацией и персоналом\n"
        "- R629 Управление маркетингом и финансами\n\n"
        f"В базе: {stats.get('books', 0)} книг, {stats.get('chapters', 0)} глав.\n\n"
        "Задайте вопрос по материалам, и я найду ответ.\n\n"
        "/help - как задавать вопросы\n"
        "/status - статус базы знаний"
    )
    await update.message.reply_text(text)


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = (
        "Как задавать вопросы:\n\n"
        "- Какие роли выполняет менеджер?\n"
        "- Что такое делегирование?\n"
        "- Что такое 4P маркетинга?\n"
        "- Как рассчитать точку безубыточности?\n\n"
        "После ответа нажмите Подробнее для изучения глав."
    )
    await update.message.reply_text(text)


async def status_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    stats = searcher.tree.get_stats() if searcher else {}
    if stats:
        text = (
            "База знаний активна.\n\n"
            f"Книг: {stats.get('books', 0)}\n"
            f"Глав: {stats.get('chapters', 0)}\n"
            f"Секций: {stats.get('sections', 0)}\n"
            f"Фрагментов: {stats.get('chunks', 0)}"
        )
        await update.message.reply_text(text)
    else:
        await update.message.reply_text("База знаний недоступна.")


async def usage_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = str(update.effective_user.id)
    if ADMIN_TELEGRAM_IDS and user_id not in ADMIN_TELEGRAM_IDS:
        return
    usage = rate_limiter.get_usage_info()
    text = f"Статистика за {usage['date']}:\nЗапросов: {usage['requests_today']} из {usage['limit']}"
    await update.message.reply_text(text)


def get_chapters_split(results, answer):
    """Разделяет главы на упомянутые в ответе и дополнительные."""
    mentioned = {}
    extra = {}
    for r in results:
        key = f"{r.book_title}|{r.chapter_title}"
        book_name = get_book_display_name(r.book_title)
        ch_data = {"book": r.book_title, "chapter": r.chapter_title, "summary": r.chapter_summary}
        if book_name in answer:
            if key not in mentioned:
                mentioned[key] = ch_data
        else:
            if key not in extra and key not in mentioned:
                extra[key] = ch_data
    return list(mentioned.values()), list(extra.values())


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    question = update.message.text.strip()
    user_id = update.effective_user.id

    if not question or not searcher:
        return

    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")
    logger.info(f"Вопрос: {question[:50]}...")

    results = searcher.search(question, top_chapters=4, top_chunks=6)

    if not results:
        await update.message.reply_text("В материалах нет информации. Переформулируйте вопрос.")
        return

    context_chunks = []
    for r in results:
        context_chunks.append({
            'text': r.text,
            'metadata': {'book_title': r.book_title, 'chapter': r.chapter_title, 'section': r.section_title},
            'score': r.score
        })

    answer = llm_client.generate_answer(question, context_chunks, is_expanded_search=True)
    rate_limiter.record_request()

    # Сохраняем в кэш с ответом
    search_results_cache[user_id] = {"results": results, "question": question, "answer": answer}

    # Добавляем только те источники, которые LLM реально упомянул в ответе
    mentioned_books = set()
    for r in results:
        book_name = get_book_display_name(r.book_title)
        if book_name in answer:
            mentioned_books.add(book_name)

    # Если LLM не упомянул ни одной книги, показываем все
    if not mentioned_books:
        for r in results:
            mentioned_books.add(get_book_display_name(r.book_title))

    sources = ", ".join(sorted(mentioned_books))
    nl = chr(10)
    book_emoji = chr(128218)
    answer_with_sources = answer + nl + nl + book_emoji + " *Источники:* " + sources

    keyboard = [[InlineKeyboardButton("Подробнее", callback_data="details")]]
    reply_markup = InlineKeyboardMarkup(keyboard)

    await update.message.reply_text(answer_with_sources, parse_mode="Markdown", reply_markup=reply_markup)


async def handle_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()

    user_id = query.from_user.id
    data = query.data

    # Игнорируем нажатие на разделитель
    if data == "noop":
        return

    cached = search_results_cache.get(user_id)
    if not cached:
        await query.edit_message_reply_markup(reply_markup=None)
        return

    results = cached["results"]
    question = cached["question"]
    answer = cached.get("answer", "")

    if data == "details":
        mentioned_chapters, extra_chapters = get_chapters_split(results, answer)

        keyboard = []

        # Добавляем упомянутые главы
        if mentioned_chapters:
            keyboard.append([InlineKeyboardButton("--- Главы из ответа ---", callback_data="noop")])
            for i, ch in enumerate(mentioned_chapters[:3]):
                book_name = get_book_display_name(ch["book"])
                ch_short = ch["chapter"][:30] + "..." if len(ch["chapter"]) > 30 else ch["chapter"]
                keyboard.append([InlineKeyboardButton(f"{book_name}: {ch_short}", callback_data=f"m_{i}")])

        # Добавляем дополнительные главы
        if extra_chapters:
            keyboard.append([InlineKeyboardButton("--- Ещё найдено ---", callback_data="noop")])
            for i, ch in enumerate(extra_chapters[:3]):
                book_name = get_book_display_name(ch["book"])
                ch_short = ch["chapter"][:30] + "..." if len(ch["chapter"]) > 30 else ch["chapter"]
                keyboard.append([InlineKeyboardButton(f"{book_name}: {ch_short}", callback_data=f"e_{i}")])

        keyboard.append([InlineKeyboardButton("Закрыть", callback_data="close")])

        # Отправляем новое сообщение, не удаляя старое
        await query.message.reply_text("Выберите главу:", reply_markup=InlineKeyboardMarkup(keyboard))

    elif data.startswith("m_") or data.startswith("e_"):
        # m_ = mentioned, e_ = extra
        is_mentioned = data.startswith("m_")
        idx = int(data[2:])

        mentioned_chapters, extra_chapters = get_chapters_split(results, answer)
        chapters_list = mentioned_chapters if is_mentioned else extra_chapters

        if idx < len(chapters_list):
            ch = chapters_list[idx]
            book_name = get_book_display_name(ch["book"])
            summary = ch["summary"] or "Краткое содержание недоступно."
            text = "*" + book_name + "*\n*" + ch["chapter"] + "*\n\n" + summary

            keyboard = [
                [InlineKeyboardButton("Другая глава", callback_data="details")],
                [InlineKeyboardButton("Закрыть", callback_data="close")]
            ]
            await query.message.reply_text(text, parse_mode="Markdown", reply_markup=InlineKeyboardMarkup(keyboard))

    elif data.startswith("ch_"):
        # Старый формат для обратной совместимости
        idx = int(data.replace("ch_", ""))
        unique_chapters = {}
        for r in results:
            key = f"{r.book_title}|{r.chapter_title}"
            if key not in unique_chapters:
                unique_chapters[key] = {"book": r.book_title, "chapter": r.chapter_title, "summary": r.chapter_summary}

        chapters_list = list(unique_chapters.values())
        if idx < len(chapters_list):
            ch = chapters_list[idx]
            book_name = get_book_display_name(ch["book"])
            summary = ch["summary"] or "Краткое содержание недоступно."
            text = "*" + book_name + "*\n*" + ch["chapter"] + "*\n\n" + summary

            keyboard = [
                [InlineKeyboardButton("Другая глава", callback_data="details")],
                [InlineKeyboardButton("Закрыть", callback_data="close")]
            ]
            await query.message.reply_text(text, parse_mode="Markdown", reply_markup=InlineKeyboardMarkup(keyboard))

    elif data == "close":
        await query.edit_message_reply_markup(reply_markup=None)
