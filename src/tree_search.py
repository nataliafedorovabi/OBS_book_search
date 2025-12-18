"""
Поиск по дереву контекста.

Стратегия поиска:
1. Переосмыслить вопрос (понять о чём он)
2. Найти релевантные главы по summary
3. Найти релевантные секции внутри глав
4. Вернуть лучшие чанки
"""
import json
import logging
import re
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from src.chapters import CHAPTERS_INFO

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """Результат поиска."""
    chunk_id: str
    text: str
    score: float
    book_title: str
    chapter_title: str
    chapter_summary: str
    section_title: str
    keywords: List[str]


class ContextTree:
    """Дерево контекста для поиска."""

    def __init__(self, tree_path: Path = None):
        self.tree_path = tree_path or Path(__file__).parent.parent / "data" / "context_tree.json"
        self.tree = None
        self._chapter_index = {}  # chapter_id -> chapter data
        self._section_index = {}  # section_id -> section data
        self._chunk_index = {}    # chunk_id -> chunk data

    def load(self) -> bool:
        """Загружает дерево из файла."""
        if not self.tree_path.exists():
            logger.warning(f"Файл дерева не найден: {self.tree_path}")
            return False

        try:
            with open(self.tree_path, 'r', encoding='utf-8') as f:
                self.tree = json.load(f)

            # Строим индексы
            self._build_indexes()

            logger.info(f"Дерево загружено: {len(self.tree.get('books', []))} книг, "
                       f"{len(self._chapter_index)} глав, "
                       f"{len(self._chunk_index)} чанков")
            return True

        except Exception as e:
            logger.error(f"Ошибка загрузки дерева: {e}")
            return False

    def _build_indexes(self):
        """Строит индексы для быстрого доступа."""
        for book in self.tree.get('books', []):
            book_title = book.get('title', '')

            for chapter in book.get('chapters', []):
                ch_id = chapter.get('id')
                self._chapter_index[ch_id] = {
                    **chapter,
                    'book_title': book_title
                }

                for section in chapter.get('sections', []):
                    sec_id = section.get('id')
                    self._section_index[sec_id] = {
                        **section,
                        'book_title': book_title,
                        'chapter_id': ch_id,
                        'chapter_title': chapter.get('title'),
                        'chapter_summary': chapter.get('summary')
                    }

                    for chunk in section.get('chunks', []):
                        chunk_id = chunk.get('id')
                        self._chunk_index[chunk_id] = {
                            **chunk,
                            'book_title': book_title,
                            'chapter_id': ch_id,
                            'chapter_title': chapter.get('title'),
                            'chapter_summary': chapter.get('summary'),
                            'section_id': sec_id,
                            'section_title': section.get('title'),
                            'section_summary': section.get('summary')
                        }

    def get_all_chapters(self) -> List[Dict]:
        """Возвращает все главы с их summary."""
        return [
            {
                'id': ch_id,
                'number': ch.get('number'),
                'title': ch.get('title'),
                'summary': ch.get('summary'),
                'key_concepts': ch.get('key_concepts', []),
                'book_title': ch.get('book_title')
            }
            for ch_id, ch in self._chapter_index.items()
        ]

    def get_chapter_chunks(self, chapter_id: str) -> List[Dict]:
        """Возвращает все чанки из главы."""
        return [
            chunk for chunk_id, chunk in self._chunk_index.items()
            if chunk.get('chapter_id') == chapter_id
        ]

    def get_section_chunks(self, section_id: str) -> List[Dict]:
        """Возвращает все чанки из секции."""
        return [
            chunk for chunk_id, chunk in self._chunk_index.items()
            if chunk.get('section_id') == section_id
        ]

    def search_chapters_by_keywords(self, keywords: List[str]) -> List[Dict]:
        """
        Ищет главы, где в summary или key_concepts есть ключевые слова.
        Возвращает отсортированный по релевантности список.
        """
        results = []

        for ch_id, ch in self._chapter_index.items():
            score = 0
            summary_lower = ch.get('summary', '').lower()
            concepts = [c.lower() for c in ch.get('key_concepts', [])]
            title_lower = ch.get('title', '').lower()

            for kw in keywords:
                kw_lower = kw.lower()
                # Вес за нахождение в разных местах
                if kw_lower in title_lower:
                    score += 3
                if kw_lower in summary_lower:
                    score += 2
                if any(kw_lower in c for c in concepts):
                    score += 2

            if score > 0:
                results.append({
                    **ch,
                    'relevance_score': score
                })

        # Сортируем по релевантности
        results.sort(key=lambda x: x['relevance_score'], reverse=True)
        return results

    def search_chunks_in_chapter(self, chapter_id: str, keywords: List[str]) -> List[Dict]:
        """Ищет чанки внутри главы по ключевым словам."""
        chapter_chunks = self.get_chapter_chunks(chapter_id)
        results = []

        for chunk in chapter_chunks:
            score = 0
            text_lower = chunk.get('text', '').lower()
            chunk_keywords = [k.lower() for k in chunk.get('keywords', [])]

            for kw in keywords:
                kw_lower = kw.lower()
                if kw_lower in text_lower:
                    score += 1
                    # Бонус за количество вхождений
                    score += text_lower.count(kw_lower) * 0.1
                if kw_lower in chunk_keywords:
                    score += 0.5

            if score > 0:
                results.append({
                    **chunk,
                    'relevance_score': score
                })

        results.sort(key=lambda x: x['relevance_score'], reverse=True)
        return results

    def get_stats(self) -> Dict:
        """Статистика дерева."""
        if not self.tree:
            return {}

        return {
            'version': self.tree.get('version'),
            'created_at': self.tree.get('created_at'),
            'books': len(self.tree.get('books', [])),
            'chapters': len(self._chapter_index),
            'sections': len(self._section_index),
            'chunks': len(self._chunk_index)
        }


class TreeSearcher:
    """
    Поисковик по дереву контекста.

    Стратегия:
    1. LLM переосмысливает вопрос → ключевые слова и темы
    2. Ищем главы по summary (быстрый keyword поиск)
    3. Внутри релевантных глав ищем чанки
    4. Возвращаем лучшие результаты с контекстом
    """

    def __init__(self, tree: ContextTree, llm_client=None):
        self.tree = tree
        self.llm = llm_client

    def search(self, query: str, top_chapters: int = 3, top_chunks: int = 5) -> List[SearchResult]:
        """
        Главный метод поиска.

        Args:
            query: Вопрос пользователя
            top_chapters: Сколько глав рассматривать
            top_chunks: Сколько чанков вернуть

        Returns:
            Список SearchResult
        """
        # Шаг 1: Понимаем вопрос
        keywords, recommended_chapters = self._understand_query(query)
        logger.info(f"Ключевые слова для поиска: {keywords}")

        # Извлекаем номера рекомендованных глав для бонуса
        recommended_nums = set()
        for ch in recommended_chapters:
            match = re.search(r'Глава\s+(\d+)', ch)
            if match:
                recommended_nums.add(int(match.group(1)))

        # Шаг 2: Ищем по ВСЕМ чанкам (не только по выбранным главам)
        all_chunks = []

        # Расширяем ключевые слова
        expanded_keywords = set()
        for kw in keywords:
            kw_lower = kw.lower()
            # Разбиваем фразы на отдельные слова
            words = kw_lower.split()
            for word in words:
                if len(word) > 2:
                    expanded_keywords.add(word)
                    # Убираем типичные русские окончания
                    for ending in ['а', 'я', 'ы', 'и', 'у', 'ю', 'ой', 'ей', 'ом', 'ем', 'ов', 'ев', 'ами', 'ями', 'ость', 'ённость']:
                        if word.endswith(ending) and len(word) > len(ending) + 2:
                            expanded_keywords.add(word[:-len(ending)])

        logger.info(f"Расширенные ключевые слова: {expanded_keywords}")

        for chunk_id, chunk in self.tree._chunk_index.items():
            score = 0
            text_lower = chunk.get('text', '').lower()
            chunk_keywords = [k.lower() for k in chunk.get('keywords', [])]

            for kw in expanded_keywords:
                if kw in text_lower:
                    # Считаем вхождения
                    count = text_lower.count(kw)
                    score += count * 1.0
                if kw in chunk_keywords:
                    score += 0.5

            if score > 0:
                # Бонус за рекомендованные главы от LLM
                chapter_title = chunk.get('chapter_title', '')
                chapter_match = re.search(r'Глава\s+(\d+)', chapter_title)
                if chapter_match and int(chapter_match.group(1)) in recommended_nums:
                    score *= 2  # Удваиваем score для рекомендованных глав

                all_chunks.append(SearchResult(
                    chunk_id=chunk_id,
                    text=chunk['text'],
                    score=score,
                    book_title=chunk.get('book_title', ''),
                    chapter_title=chapter_title,
                    chapter_summary=chunk.get('chapter_summary', ''),
                    section_title=chunk.get('section_title', ''),
                    keywords=chunk.get('keywords', [])
                ))

        # Сортируем по score
        all_chunks.sort(key=lambda x: x.score, reverse=True)

        # Берём топ, но стараемся взять из разных глав
        result = []
        chapters_count = {}  # глава -> сколько чанков взяли

        for chunk in all_chunks:
            if len(result) >= top_chunks:
                break
            # Берём не более 2 чанков из одной главы
            ch = chunk.chapter_title
            if chapters_count.get(ch, 0) < 2:
                result.append(chunk)
                chapters_count[ch] = chapters_count.get(ch, 0) + 1

        logger.info(f"Найдено: {len(result)} чанков из {len(chapters_count)} глав")
        return result

    def _understand_query(self, query: str) -> tuple:
        """
        Переосмысливает запрос, извлекая ключевые слова и рекомендуемые главы.
        Использует CHAPTERS_INFO с подсказками о связях терминов.

        Returns:
            (keywords, recommended_chapters)
        """
        if self.llm:
            analysis = self.llm.understand_query(query, CHAPTERS_INFO)
            keywords = analysis.get('search_terms', [])
            chapters = analysis.get('chapters', [])
            if keywords:
                logger.info(f"LLM понял запрос: термины={keywords}, главы={[c[:20] for c in chapters]}")
                return keywords, chapters

        # Fallback: простое разбиение на слова
        import re
        words = re.findall(r'[а-яА-ЯёЁa-zA-Z]+', query.lower())
        stop_words = {'что', 'как', 'где', 'когда', 'почему', 'какой', 'какая', 'какие',
                     'это', 'такое', 'для', 'при', 'или', 'если', 'чем', 'между'}
        keywords = [w for w in words if w not in stop_words and len(w) > 2]
        return keywords[:5], []

    def get_chapters_info(self) -> str:
        """Возвращает информацию о главах для промпта LLM."""
        chapters = self.tree.get_all_chapters()
        lines = []
        for ch in chapters:
            concepts = ', '.join(ch.get('key_concepts', []))
            lines.append(f"Глава {ch['number']}. {ch['title']}")
            if ch.get('summary'):
                lines.append(f"  {ch['summary'][:200]}...")
            if concepts:
                lines.append(f"  Ключевые концепции: {concepts}")
            lines.append("")
        return "\n".join(lines)
