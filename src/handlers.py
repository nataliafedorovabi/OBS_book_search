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


def pluralize(n: int, form1: str, form2: str, form5: str) -> str:
    """Склонение слова по числу: 1 книга, 2 книги, 5 книг."""
    n = abs(n)
    if n % 10 == 1 and n % 100 != 11:
        return form1
    elif 2 <= n % 10 <= 4 and not (12 <= n % 100 <= 14):
        return form2
    else:
        return form5


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
    books = stats.get("books", 0)
    chapters = stats.get("chapters", 0)
    text = "Привет! Я бот-ассистент курсов:" + nl
    text += "- R628 Управление организацией и персоналом" + nl
    text += "- R629 Управление маркетингом и финансами" + nl + nl
    text += "В базе: " + str(books) + " " + pluralize(books, "книга", "книги", "книг")
    text += ", " + str(chapters) + " " + pluralize(chapters, "глава", "главы", "глав") + "." + nl + nl
    text += "Задайте вопрос по материалам, и я найду ответ." + nl + nl
    text += "/help - как задавать вопросы" + nl
    text += "/status - статус базы знаний"

    user_id = str(update.effective_user.id)
    logger.info(f"start_command: user_id={user_id}, ADMIN_IDS={ADMIN_TELEGRAM_IDS}")
    if ADMIN_TELEGRAM_IDS and user_id in ADMIN_TELEGRAM_IDS:
        text += nl + "/usage - статистика (админ)"

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
        books = stats.get("books", 0)
        chapters = stats.get("chapters", 0)
        sections = stats.get("sections", 0)
        chunks = stats.get("chunks", 0)
        text = "База знаний активна." + nl + nl
        text += str(books) + " " + pluralize(books, "книга", "книги", "книг") + nl
        text += str(chapters) + " " + pluralize(chapters, "глава", "главы", "глав") + nl
        text += str(sections) + " " + pluralize(sections, "секция", "секции", "секций") + nl
        text += str(chunks) + " " + pluralize(chunks, "фрагмент", "фрагмента", "фрагментов")
        await update.message.reply_text(text)
    else:
        await update.message.reply_text("База знаний недоступна.")


async def usage_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = str(update.effective_user.id)
    if ADMIN_TELEGRAM_IDS and user_id not in ADMIN_TELEGRAM_IDS:
        return

    stats = rate_limiter.get_admin_stats()
    nl = chr(10)

    # Header
    text = "=== Статистика бота ===" + nl + nl

    # Today
    reqs_today = stats["requests_today"]
    text += "Сегодня (" + stats["date"] + "):" + nl
    text += str(reqs_today) + " " + pluralize(reqs_today, "запрос", "запроса", "запросов")
    text += " из " + str(stats["limit"]) + nl + nl

    # All time
    total = stats["total_requests"]
    days = stats["days_tracked"]
    avg = stats["avg_per_day"]
    text += "Всего:" + nl
    text += str(total) + " " + pluralize(total, "запрос", "запроса", "запросов") + nl
    text += str(days) + " " + pluralize(days, "день", "дня", "дней") + " активности" + nl
    text += "~" + str(avg) + " " + pluralize(int(avg), "запрос", "запроса", "запросов") + "/день" + nl + nl

    # Users
    users_count = stats["total_users"]
    text += str(users_count) + " " + pluralize(users_count, "пользователь", "пользователя", "пользователей") + nl

    await update.message.reply_text(text)

    # Send user details as separate messages
    for u in stats["users"][:10]:
        utext = "--- " + u["name"]
        if u["username"]:
            utext += " (@" + u["username"] + ")"
        utext += " ---" + nl
        utext += "ID: " + u["user_id"] + nl
        utext += "Всего: " + str(u["total_requests"]) + " " + pluralize(u["total_requests"], "запрос", "запроса", "запросов") + nl
        utext += "Сегодня: " + str(u["requests_today"]) + nl + nl

        if u["recent_questions"]:
            utext += "Последние вопросы:" + nl
            for q in u["recent_questions"]:
                if q:
                    short_q = (q[:80] + "...") if len(q) > 80 else q
                    utext += "• " + short_q + nl

        await update.message.reply_text(utext)


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

    user = update.effective_user
    user_info = {
        'first_name': user.first_name or '',
        'last_name': user.last_name or '',
        'username': user.username or ''
    }
    rate_limiter.record_request(user_id=user_id, user_info=user_info, question=question)

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
