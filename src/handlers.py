import logging
import re
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
    nl = chr(10)
    text = "Привет! Я бот-ассистент курсов:" + nl
    text += "- R628 Управление организацией и персоналом" + nl
    text += "- R629 Управление маркетингом и финансами" + nl + nl
    text += "В базе: " + str(stats.get("books", 0)) + " книг, " + str(stats.get("chapters", 0)) + " глав." + nl + nl
    text += "Задайте вопрос по материалам, и я найду ответ." + nl + nl
    text += "/help - как задавать вопросы" + nl
    text += "/status - статус базы знаний"
    await update.message.reply_text(text)


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    nl = chr(10)
    text = "Как задавать вопросы:" + nl + nl
    text += "- Какие роли выполняет менеджер?" + nl
    text += "- Что такое делегирование?" + nl
    text += "- Что такое 4P маркетинга?" + nl
    text += "- Как рассчитать точку безубыточности?" + nl + nl
    text += "После ответа нажмите Подробнее для изучения глав."
    await update.message.reply_text(text)


async def status_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    stats = searcher.tree.get_stats() if searcher else {}
    nl = chr(10)
    if stats:
        text = "База знаний активна." + nl + nl
        text += "Книг: " + str(stats.get("books", 0)) + nl
        text += "Глав: " + str(stats.get("chapters", 0)) + nl
        text += "Секций: " + str(stats.get("sections", 0)) + nl
        text += "Фрагментов: " + str(stats.get("chunks", 0))
        await update.message.reply_text(text)
    else:
        await update.message.reply_text("База знаний недоступна.")


async def usage_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = str(update.effective_user.id)
    if ADMIN_TELEGRAM_IDS and user_id not in ADMIN_TELEGRAM_IDS:
        return
    usage = rate_limiter.get_usage_info()
    nl = chr(10)
    text = "Статистика за " + usage["date"] + ":" + nl + "Запросов: " + str(usage["requests_today"]) + " из " + str(usage["limit"])
    await update.message.reply_text(text)


def get_mentioned_chapters(results, answer):
    mentioned = []
    seen_keys = set()

    for r in results:
        key = r.book_title + "|" + r.chapter_title
        if key in seen_keys:
            continue

        book_name = get_book_display_name(r.book_title)

        ch_match = re.search(r"Глава\s*(\d+)", r.chapter_title)
        if not ch_match:
            continue
        ch_num = ch_match.group(1)

        pattern = re.escape(book_name) + r".*?Глава\s*" + ch_num + r"\b"
        if re.search(pattern, answer, re.DOTALL):
            seen_keys.add(key)
            mentioned.append({"book": r.book_title, "chapter": r.chapter_title, "summary": r.chapter_summary})

    return mentioned


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    question = update.message.text.strip()
    user_id = update.effective_user.id

    if not question or not searcher:
        return

    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")
    logger.info("Вопрос: " + question[:50] + "...")

    results = searcher.search(question, top_chapters=4, top_chunks=6)

    if not results:
        await update.message.reply_text("В материалах нет информации. Переформулируйте вопрос.")
        return

    context_chunks = []
    for r in results:
        context_chunks.append({
            "text": r.text,
            "metadata": {"book_title": r.book_title, "chapter": r.chapter_title, "section": r.section_title},
            "score": r.score
        })

    answer = llm_client.generate_answer(question, context_chunks, is_expanded_search=True)
    rate_limiter.record_request()

    search_results_cache[user_id] = {"results": results, "question": question, "answer": answer}

    mentioned_books = set()
    for r in results:
        book_name = get_book_display_name(r.book_title)
        if book_name in answer:
            mentioned_books.add(book_name)

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

    cached = search_results_cache.get(user_id)
    if not cached:
        await query.edit_message_reply_markup(reply_markup=None)
        return

    results = cached["results"]
    answer = cached.get("answer", "")
    nl = chr(10)

    if data == "details":
        chapters = get_mentioned_chapters(results, answer)

        if not chapters:
            await query.message.reply_text("Главы не найдены.")
            return

        keyboard = []
        for i, ch in enumerate(chapters[:6]):
            book_name = get_book_display_name(ch["book"])
            ch_title = ch["chapter"]
            ch_short = ch_title[:30] + "..." if len(ch_title) > 30 else ch_title
            keyboard.append([InlineKeyboardButton(book_name + ": " + ch_short, callback_data="ch_" + str(i))])

        keyboard.append([InlineKeyboardButton("Закрыть", callback_data="close")])

        await query.message.reply_text("Выберите главу:", reply_markup=InlineKeyboardMarkup(keyboard))

    elif data.startswith("ch_"):
        idx = int(data.replace("ch_", ""))
        chapters = get_mentioned_chapters(results, answer)

        if idx < len(chapters):
            ch = chapters[idx]
            book_name = get_book_display_name(ch["book"])
            summary = ch["summary"] or "Краткое содержание недоступно."
            header = "*" + book_name + "*" + nl + "*" + ch["chapter"] + "*" + nl + nl

            keyboard = [
                [InlineKeyboardButton("Другая глава", callback_data="details")],
                [InlineKeyboardButton("Закрыть", callback_data="close")]
            ]
            try:
                await query.message.reply_text(header + summary, parse_mode="Markdown", reply_markup=InlineKeyboardMarkup(keyboard))
            except Exception as e:
                logger.warning("Markdown error: " + str(e))
                # Очищаем Markdown разметку для plain текста
                clean_summary = re.sub(r'\*+', '', summary)
                clean_summary = re.sub(r'_+', '', clean_summary)
                plain = book_name + nl + ch["chapter"] + nl + nl + clean_summary
                await query.message.reply_text(plain, reply_markup=InlineKeyboardMarkup(keyboard))

    elif data == "close":
        await query.edit_message_reply_markup(reply_markup=None)
