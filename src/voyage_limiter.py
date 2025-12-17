"""
Лимитер для Voyage AI - защита от неожиданных списаний.

voyage-multilingual-2:
- Бесплатно: 50M токенов
- После: $0.06/1M токенов
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Callable

logger = logging.getLogger(__name__)

# Настройки лимитов
FREE_TOKEN_LIMIT = 50_000_000  # 50M бесплатных токенов
WARNING_THRESHOLD = 0.8  # Предупредить при 80% использования (40M)
HARD_LIMIT = 45_000_000  # Жёсткий лимит - остановить при 45M (оставить буфер 5M)


class VoyageLimiter:
    """Отслеживает использование токенов Voyage AI и блокирует при превышении."""

    def __init__(self, data_dir: Path):
        self.stats_file = data_dir / "voyage_usage.json"
        self.total_tokens = 0
        self.warning_sent = False
        self.limit_reached = False
        self._notify_callback: Optional[Callable[[str], None]] = None
        self._load_stats()

    def _load_stats(self):
        """Загружает статистику из файла."""
        if self.stats_file.exists():
            try:
                with open(self.stats_file, 'r') as f:
                    data = json.load(f)
                    self.total_tokens = data.get('total_tokens', 0)
                    self.warning_sent = data.get('warning_sent', False)
                    self.limit_reached = data.get('limit_reached', False)
                logger.info(f"Voyage usage загружен: {self.total_tokens:,} токенов "
                           f"({self.total_tokens/FREE_TOKEN_LIMIT*100:.2f}% лимита)")
            except Exception as e:
                logger.error(f"Ошибка загрузки voyage_usage.json: {e}")

    def _save_stats(self):
        """Сохраняет статистику в файл."""
        try:
            self.stats_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.stats_file, 'w') as f:
                json.dump({
                    'total_tokens': self.total_tokens,
                    'warning_sent': self.warning_sent,
                    'limit_reached': self.limit_reached,
                    'last_updated': datetime.now().isoformat(),
                    'free_limit': FREE_TOKEN_LIMIT,
                    'hard_limit': HARD_LIMIT
                }, f, indent=2)
        except Exception as e:
            logger.error(f"Ошибка сохранения voyage_usage.json: {e}")

    def set_notify_callback(self, callback: Callable[[str], None]):
        """Устанавливает callback для уведомлений."""
        self._notify_callback = callback

    def can_make_request(self) -> bool:
        """Проверяет, можно ли делать запрос (не превышен ли лимит)."""
        if self.total_tokens >= HARD_LIMIT:
            if not self.limit_reached:
                self.limit_reached = True
                self._save_stats()
                self._notify(
                    f"VOYAGE AI ЛИМИТ ДОСТИГНУТ!\n"
                    f"Использовано: {self.total_tokens:,} токенов\n"
                    f"Лимит: {HARD_LIMIT:,} токенов\n"
                    f"Бот остановлен для защиты от списаний.\n"
                    f"Добавьте бюджет или сбросьте счётчик командой /voyage_reset"
                )
            return False
        return True

    def record_usage(self, tokens: int):
        """Записывает использование токенов."""
        self.total_tokens += tokens
        self._save_stats()

        percent_used = self.total_tokens / FREE_TOKEN_LIMIT * 100
        logger.info(f"Voyage: +{tokens} токенов, всего {self.total_tokens:,} ({percent_used:.2f}%)")

        # Предупреждение при 80%
        if not self.warning_sent and self.total_tokens >= FREE_TOKEN_LIMIT * WARNING_THRESHOLD:
            self.warning_sent = True
            self._save_stats()
            self._notify(
                f"Voyage AI: использовано {percent_used:.1f}% бесплатного лимита!\n"
                f"Токенов: {self.total_tokens:,} / {FREE_TOKEN_LIMIT:,}\n"
                f"Осталось до остановки: {HARD_LIMIT - self.total_tokens:,}"
            )

    def _notify(self, message: str):
        """Отправляет уведомление."""
        logger.warning(message)
        if self._notify_callback:
            try:
                self._notify_callback(message)
            except Exception as e:
                logger.error(f"Ошибка отправки уведомления: {e}")

    def get_stats(self) -> dict:
        """Возвращает статистику использования."""
        return {
            'total_tokens': self.total_tokens,
            'free_limit': FREE_TOKEN_LIMIT,
            'hard_limit': HARD_LIMIT,
            'percent_used': round(self.total_tokens / FREE_TOKEN_LIMIT * 100, 2),
            'remaining': HARD_LIMIT - self.total_tokens,
            'limit_reached': self.limit_reached,
            'warning_sent': self.warning_sent
        }

    def reset(self, admin_confirmed: bool = False) -> bool:
        """
        Сбрасывает счётчик (только с подтверждением админа).
        Использовать после пополнения баланса Voyage или начала нового периода.
        """
        if not admin_confirmed:
            return False

        old_tokens = self.total_tokens
        self.total_tokens = 0
        self.warning_sent = False
        self.limit_reached = False
        self._save_stats()

        logger.info(f"Voyage счётчик сброшен: {old_tokens:,} → 0")
        return True
