import asyncio
import logging
from typing import Dict, List, Any
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ContextTypes, Application
from src.vector_store import VectorStore, set_admin_notify_callback, get_voyage_limiter
from src.llm import LLMClient
from src.rate_limiter import RateLimiter
from src.config import ADMIN_TELEGRAM_IDS

logger = logging.getLogger(__name__)

# Глобальные объекты (инициализируются в main.py)
vector_store: VectorStore = None
llm_client: LLMClient = None
rate_limiter: RateLimiter = None
_bot_app: Application = None  # Для отправки уведомлений

# Хранение контекста поиска для кнопок (user_id -> context)
_search_context: Dict[int, Dict[str, Any]] = {}


def init_services(vs: VectorStore, llm: LLMClient, app: Application = None):
    """Инициализация сервисов."""
    global vector_store, llm_client, rate_limiter, _bot_app
    vector_store = vs
    llm_client = llm
    rate_limiter = RateLimiter()
    _bot_app = app

    # Настраиваем callback для уведомлений админа о проблемах с Voyage AI
    set_admin_notify_callback(_send_admin_notification)
    logger.info("Callback для уведомлений админа настроен")


def _send_admin_notification(message: str):
    """Синхронная обёртка для отправки уведомлений админам."""
    if not ADMIN_TELEGRAM_IDS or not _bot_app:
        logger.warning(f"Не удалось отправить уведомление: {message}")
        return

    async def _send():
        for admin_id in ADMIN_TELEGRAM_IDS:
            try:
                await _bot_app.bot.send_message(
                    chat_id=admin_id,
                    text=f"🚨 {message}"
                )
                logger.info(f"Уведомление отправлено админу {admin_id}")
            except Exception as e:
                logger.error(f"Ошибка отправки уведомления админу {admin_id}: {e}")

    # Запускаем асинхронную отправку
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            asyncio.create_task(_send())
        else:
            loop.run_until_complete(_send())
    except Exception as e:
        logger.error(f"Ошибка при отправке уведомления: {e}")


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

    # 1. Сначала обычный поиск
    relevant_chunks = vector_store.search(question)
    is_expanded = False

    # Логируем результаты первого поиска
    if relevant_chunks:
        top_scores = [f"{c.get('score', 0):.2f}" for c in relevant_chunks[:3]]
        logger.info(f"Первый поиск: {len(relevant_chunks)} чанков, scores={top_scores}")
    else:
        logger.info("Первый поиск: ничего не найдено")

    # 2. Проверяем качество результатов
    # Условие: высокий score И ключевые слова запроса есть в найденных чанках
    def check_keyword_match(query: str, chunks: list) -> bool:
        """Проверяет, содержат ли чанки ключевые слова из запроса."""
        import re
        # Извлекаем значимые слова (>4 букв, не стоп-слова)
        stop = {'найди', 'покажи', 'расскажи', 'модель', 'какой', 'какая', 'какие', 'который'}
        words = re.findall(r'[а-яёa-z]{5,}', query.lower())
        keywords = [w for w in words if w not in stop]

        if not keywords:
            return True  # Нет ключевых слов для проверки

        # Проверяем есть ли хотя бы одно ключевое слово в чанках
        all_text = ' '.join(c.get('text', '').lower() for c in chunks)
        matches = sum(1 for kw in keywords if kw in all_text)
        match_ratio = matches / len(keywords) if keywords else 0

        logger.info(f"Проверка ключевых слов: {keywords} -> совпадений {matches}/{len(keywords)} ({match_ratio:.0%})")
        return match_ratio > 0.5  # Больше половины слов должно быть

    has_good_score = relevant_chunks and any(c.get('score', 0) >= 0.5 for c in relevant_chunks)
    has_keyword_match = check_keyword_match(question, relevant_chunks) if relevant_chunks else False

    has_good_results = has_good_score and has_keyword_match

    if not has_keyword_match and has_good_score:
        logger.info("Score высокий, но ключевые слова не найдены - форсируем расширение")

    if not has_good_results:
        logger.info(f"Прямой поиск не дал хороших результатов, расширяем запрос")

        # Расширяем запрос через LLM
        expanded = llm_client.expand_query(question)
        search_terms = expanded.get('search_terms', [])
        target_chapters = expanded.get('chapters', [])

        logger.info(f"Расширение: главы={target_chapters}, термины={search_terms}")

        if search_terms and target_chapters:
            # УМНЫЙ ПОИСК: ищем в КАЖДОЙ главе отдельно
            all_chunks = {}
            chapters_found = set()

            for chapter in target_chapters:
                chapter_chunks = {}
                for term in search_terms:
                    # Ищем этот термин в конкретной главе
                    chunks = vector_store.search(term, n_results=2, chapters=[chapter])
                    for chunk in chunks:
                        chunk_id = chunk.get('metadata', {}).get('id', id(chunk))
                        if chunk_id not in chapter_chunks or chunk['score'] > chapter_chunks[chunk_id]['score']:
                            chapter_chunks[chunk_id] = chunk

                # Берём лучший результат из этой главы
                if chapter_chunks:
                    best_from_chapter = sorted(chapter_chunks.values(), key=lambda x: x['score'], reverse=True)[:2]
                    for chunk in best_from_chapter:
                        chunk_id = chunk.get('metadata', {}).get('id', id(chunk))
                        all_chunks[chunk_id] = chunk
                        chapters_found.add(chapter.split('.')[0] if '.' in chapter else chapter)

            logger.info(f"Найдено из глав: {chapters_found}")

            # Сортируем по score, берём топ результаты
            relevant_chunks = sorted(all_chunks.values(), key=lambda x: x['score'], reverse=True)[:6]
            is_expanded = True
            logger.info(f"Расширенный поиск нашёл {len(relevant_chunks)} чанков из {len(chapters_found)} глав")

        elif search_terms:
            # Fallback: если глав нет, ищем просто по терминам
            all_chunks = {}
            for term in search_terms:
                chunks = vector_store.search(term, n_results=3)
                for chunk in chunks:
                    chunk_id = chunk.get('metadata', {}).get('id', id(chunk))
                    if chunk_id not in all_chunks or chunk['score'] > all_chunks[chunk_id]['score']:
                        all_chunks[chunk_id] = chunk

            relevant_chunks = sorted(all_chunks.values(), key=lambda x: x['score'], reverse=True)[:5]
            is_expanded = True
            logger.info(f"Расширенный поиск (без глав) нашёл {len(relevant_chunks)} чанков")

    # Генерируем ответ через LLM
    answer = llm_client.generate_answer(question, relevant_chunks, is_expanded_search=is_expanded)

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

    # Группируем чанки по главам для кнопок
    chapters_in_results = {}
    for chunk in relevant_chunks:
        chapter = chunk.get('metadata', {}).get('chapter', 'Без главы')
        if chapter not in chapters_in_results:
            chapters_in_results[chapter] = []
        chapters_in_results[chapter].append(chunk)

    # Сохраняем контекст поиска
    user_id = update.effective_user.id
    _search_context[user_id] = {
        'query': question,
        'chunks': relevant_chunks,
        'chapters': chapters_in_results,
        'is_expanded': is_expanded,
        'search_depth': 1  # Уровень глубины поиска
    }

    # Создаём кнопки
    keyboard = []

    # Кнопки "Подробнее" для каждой главы (если несколько глав)
    if len(chapters_in_results) > 1:
        for i, chapter in enumerate(list(chapters_in_results.keys())[:3]):
            short_name = chapter.split('. ')[1][:20] + '...' if '. ' in chapter and len(chapter.split('. ')[1]) > 20 else chapter.split('. ')[1] if '. ' in chapter else chapter[:25]
            keyboard.append([InlineKeyboardButton(f"📖 {short_name}", callback_data=f"chapter_{i}")])

    # Кнопка "Искал другое"
    keyboard.append([InlineKeyboardButton("🔄 Искал другое", callback_data="search_other")])

    reply_markup = InlineKeyboardMarkup(keyboard) if keyboard else None

    await update.message.reply_text(answer, reply_markup=reply_markup)


async def usage_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик команды /usage - статистика использования (для админов)."""
    user_id = str(update.effective_user.id)

    # Команда только для админов
    if ADMIN_TELEGRAM_IDS and user_id not in ADMIN_TELEGRAM_IDS:
        return

    usage = rate_limiter.get_usage_info()
    voyage_stats = vector_store.get_embedding_stats() if vector_store else {}

    await update.message.reply_text(
        f"📊 Статистика за {usage['date']}:\n\n"
        f"Запросов: {usage['requests_today']}/{usage['limit']}\n"
        f"Использовано: {usage['percent_used']}%\n"
        f"Осталось: {usage['remaining']}\n\n"
        f"🔍 Voyage AI (эмбеддинги):\n"
        f"Запросов: {voyage_stats.get('request_count', 0)}\n"
        f"Токенов: {voyage_stats.get('total_tokens', 0)}\n"
        f"Модель: {voyage_stats.get('model', 'N/A')}"
    )


async def voyage_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик команды /voyage - статистика Voyage AI (для админов)."""
    user_id = str(update.effective_user.id)

    # Команда только для админов
    if ADMIN_TELEGRAM_IDS and user_id not in ADMIN_TELEGRAM_IDS:
        return

    limiter = get_voyage_limiter()
    if not limiter:
        await update.message.reply_text("Voyage лимитер не инициализирован.")
        return

    stats = limiter.get_stats()

    status_emoji = "🟢" if stats['percent_used'] < 80 else "🟡" if stats['percent_used'] < 90 else "🔴"
    blocked_status = "⛔ ЗАБЛОКИРОВАН" if stats['limit_reached'] else "✅ Активен"

    await update.message.reply_text(
        f"📊 Voyage AI Статистика:\n\n"
        f"Статус: {blocked_status}\n"
        f"{status_emoji} Использовано: {stats['percent_used']:.2f}%\n\n"
        f"Токенов: {stats['total_tokens']:,} / {stats['free_limit']:,}\n"
        f"Жёсткий лимит: {stats['hard_limit']:,}\n"
        f"Осталось до блокировки: {stats['remaining']:,}\n\n"
        f"⚠️ Предупреждение отправлено: {'Да' if stats['warning_sent'] else 'Нет'}\n\n"
        f"Сбросить счётчик: /voyage_reset"
    )


async def voyage_reset_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик команды /voyage_reset - сброс счётчика (для админов)."""
    user_id = str(update.effective_user.id)

    # Команда только для админов
    if ADMIN_TELEGRAM_IDS and user_id not in ADMIN_TELEGRAM_IDS:
        return

    limiter = get_voyage_limiter()
    if not limiter:
        await update.message.reply_text("Voyage лимитер не инициализирован.")
        return

    # Проверяем аргумент подтверждения
    args = context.args
    if args and args[0] == "CONFIRM":
        stats_before = limiter.get_stats()
        if limiter.reset(admin_confirmed=True):
            await update.message.reply_text(
                f"✅ Счётчик Voyage AI сброшен!\n\n"
                f"Было: {stats_before['total_tokens']:,} токенов\n"
                f"Стало: 0 токенов\n\n"
                f"Бот снова принимает запросы."
            )
            logger.info(f"Voyage счётчик сброшен админом {user_id}")
        else:
            await update.message.reply_text("❌ Ошибка сброса счётчика.")
    else:
        stats = limiter.get_stats()
        await update.message.reply_text(
            f"⚠️ Вы уверены, что хотите сбросить счётчик?\n\n"
            f"Текущее использование: {stats['total_tokens']:,} токенов\n\n"
            f"Это следует делать только если:\n"
            f"• Начался новый период (месяц)\n"
            f"• Вы пополнили баланс Voyage AI\n\n"
            f"Для подтверждения: /voyage_reset CONFIRM"
        )


async def button_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик нажатий на inline кнопки."""
    query = update.callback_query
    await query.answer()

    user_id = update.effective_user.id
    search_ctx = _search_context.get(user_id)

    if not search_ctx:
        await query.edit_message_reply_markup(reply_markup=None)
        await query.message.reply_text("Контекст поиска устарел. Задайте вопрос заново.")
        return

    callback_data = query.data

    if callback_data.startswith("chapter_"):
        # Нажали на кнопку главы - показываем подробнее
        chapter_idx = int(callback_data.split("_")[1])
        chapters_list = list(search_ctx['chapters'].keys())

        if chapter_idx >= len(chapters_list):
            await query.message.reply_text("Глава не найдена.")
            return

        chapter_name = chapters_list[chapter_idx]
        chapter_chunks = search_ctx['chapters'][chapter_name]

        # Ищем больше информации в этой главе
        original_query = search_ctx['query']
        logger.info(f"Углублённый поиск в главе: {chapter_name}")

        # Показываем статус
        await context.bot.send_chat_action(
            chat_id=update.effective_chat.id,
            action="typing"
        )

        # Ищем дополнительные чанки из этой главы
        more_chunks = vector_store.search(original_query, n_results=5, chapters=[chapter_name])

        if more_chunks:
            # Генерируем расширенный ответ по главе
            detailed_answer = llm_client.generate_answer(
                f"{original_query} (подробнее из главы '{chapter_name}')",
                more_chunks,
                is_expanded_search=True
            )
            rate_limiter.record_request()

            await query.message.reply_text(
                f"📖 *Подробнее из {chapter_name}:*\n\n{detailed_answer}",
                parse_mode="Markdown"
            )
        else:
            await query.message.reply_text(f"Дополнительной информации в главе '{chapter_name}' не найдено.")

    elif callback_data == "search_other":
        # Нажали "Искал другое" - углубляем поиск
        original_query = search_ctx['query']
        search_depth = search_ctx.get('search_depth', 1)

        if search_depth >= 3:
            await query.message.reply_text(
                "Поиск уже максимально расширен. Попробуйте переформулировать вопрос или задать его иначе."
            )
            return

        logger.info(f"Расширяем поиск, глубина: {search_depth + 1}")

        await context.bot.send_chat_action(
            chat_id=update.effective_chat.id,
            action="typing"
        )

        # Принудительно запускаем расширенный поиск
        expanded = llm_client.expand_query(original_query)
        search_terms = expanded.get('search_terms', [])
        target_chapters = expanded.get('chapters', [])

        # Ищем с большим количеством результатов
        all_chunks = {}
        for chapter in target_chapters:
            for term in search_terms:
                chunks = vector_store.search(term, n_results=4, chapters=[chapter])
                for chunk in chunks:
                    chunk_id = chunk.get('metadata', {}).get('id', id(chunk))
                    if chunk_id not in all_chunks or chunk['score'] > all_chunks[chunk_id]['score']:
                        all_chunks[chunk_id] = chunk

        # Исключаем уже показанные чанки
        shown_ids = {c.get('metadata', {}).get('id') for c in search_ctx['chunks']}
        new_chunks = [c for c in all_chunks.values() if c.get('metadata', {}).get('id') not in shown_ids]
        new_chunks = sorted(new_chunks, key=lambda x: x['score'], reverse=True)[:5]

        if new_chunks:
            answer = llm_client.generate_answer(original_query, new_chunks, is_expanded_search=True)
            rate_limiter.record_request()

            # Обновляем контекст
            _search_context[user_id]['chunks'].extend(new_chunks)
            _search_context[user_id]['search_depth'] = search_depth + 1

            # Кнопка для ещё одного расширения
            keyboard = [[InlineKeyboardButton("🔄 Ещё варианты", callback_data="search_other")]]
            reply_markup = InlineKeyboardMarkup(keyboard)

            await query.message.reply_text(
                f"🔍 *Дополнительные результаты:*\n\n{answer}",
                parse_mode="Markdown",
                reply_markup=reply_markup
            )
        else:
            await query.message.reply_text(
                "Других релевантных материалов не найдено. Попробуйте переформулировать вопрос."
            )
