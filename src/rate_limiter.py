"""Rate limiter for API requests."""
import json
from datetime import datetime, date
from pathlib import Path
from src.config import DAILY_REQUEST_LIMIT, WARNING_THRESHOLD, DATA_DIR


STATS_FILE = DATA_DIR / "usage_stats.json"


class RateLimiter:
    def __init__(self):
        self.today = str(date.today())
        self.requests_today = 0
        self.warning_sent = False
        self._load_stats()

    def _load_stats(self):
        """Load stats from file."""
        if STATS_FILE.exists():
            try:
                with open(STATS_FILE, 'r') as f:
                    data = json.load(f)
                    if data.get('date') == self.today:
                        self.requests_today = data.get('requests', 0)
                        self.warning_sent = data.get('warning_sent', False)
            except:
                pass

    def _save_stats(self):
        """Save stats to file."""
        try:
            with open(STATS_FILE, 'w') as f:
                json.dump({
                    'date': self.today,
                    'requests': self.requests_today,
                    'warning_sent': self.warning_sent
                }, f)
        except:
            pass

    def _reset_if_new_day(self):
        """Reset counters if new day."""
        today = str(date.today())
        if today != self.today:
            self.today = today
            self.requests_today = 0
            self.warning_sent = False

    def can_make_request(self) -> bool:
        """Check if request is allowed."""
        self._reset_if_new_day()
        return self.requests_today < DAILY_REQUEST_LIMIT

    def record_request(self):
        """Record a request."""
        self._reset_if_new_day()
        self.requests_today += 1
        self._save_stats()

    def should_warn_admin(self) -> bool:
        """Check if admin should be warned."""
        if self.warning_sent:
            return False
        threshold = int(DAILY_REQUEST_LIMIT * WARNING_THRESHOLD)
        return self.requests_today >= threshold

    def mark_warning_sent(self):
        """Mark that warning was sent."""
        self.warning_sent = True
        self._save_stats()

    def get_usage_info(self) -> dict:
        """Get current usage stats."""
        self._reset_if_new_day()
        return {
            'date': self.today,
            'requests_today': self.requests_today,
            'limit': DAILY_REQUEST_LIMIT,
            'remaining': DAILY_REQUEST_LIMIT - self.requests_today,
            'percent_used': round(self.requests_today / DAILY_REQUEST_LIMIT * 100, 1)
        }
