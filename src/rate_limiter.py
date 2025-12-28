"""Rate limiter for API requests with user tracking."""
import json
from datetime import datetime, date, timedelta
from pathlib import Path
from src.config import DAILY_REQUEST_LIMIT, WARNING_THRESHOLD, STATS_DIR

# Создаём директорию для статистики если её нет (для Railway Volume)
STATS_DIR.mkdir(parents=True, exist_ok=True)
STATS_FILE = STATS_DIR / "usage_stats.json"


class RateLimiter:
    def __init__(self):
        self.today = str(date.today())
        self.requests_today = 0
        self.warning_sent = False
        self.daily_stats = {}
        self.users = {}
        self._load_stats()

    def _load_stats(self):
        """Load stats from file."""
        if STATS_FILE.exists():
            try:
                with open(STATS_FILE, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.daily_stats = data.get('daily_stats', {})
                    self.users = data.get('users', {})

                    if self.today in self.daily_stats:
                        day_data = self.daily_stats[self.today]
                        self.requests_today = day_data.get('requests', 0)
                        self.warning_sent = day_data.get('warning_sent', False)

                    # Migration from old format
                    if 'date' in data and 'requests' in data:
                        old_date = data['date']
                        if old_date not in self.daily_stats:
                            self.daily_stats[old_date] = {'requests': data['requests']}
                        if old_date == self.today:
                            self.requests_today = data['requests']
                            self.warning_sent = data.get('warning_sent', False)
            except:
                pass

    def _save_stats(self):
        """Save stats to file."""
        try:
            self.daily_stats[self.today] = {
                'requests': self.requests_today,
                'warning_sent': self.warning_sent
            }
            with open(STATS_FILE, 'w', encoding='utf-8') as f:
                json.dump({
                    'daily_stats': self.daily_stats,
                    'users': self.users
                }, f, ensure_ascii=False, indent=2)
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

    def record_request(self, user_id: int = None, user_info: dict = None, question: str = None):
        """Record a request with optional user tracking."""
        self._reset_if_new_day()
        self.requests_today += 1

        if user_id is not None:
            uid = str(user_id)
            now = datetime.now()

            if uid not in self.users:
                self.users[uid] = {
                    'first_name': '',
                    'last_name': '',
                    'username': '',
                    'first_seen': now.isoformat(),
                    'requests': []
                }

            if user_info:
                self.users[uid]['first_name'] = user_info.get('first_name', '')
                self.users[uid]['last_name'] = user_info.get('last_name', '')
                self.users[uid]['username'] = user_info.get('username', '')

            self.users[uid]['last_seen'] = now.isoformat()
            self.users[uid]['requests'].append({
                'date': self.today,
                'time': now.strftime('%H:%M:%S'),
                'question': (question[:200] + '...') if question and len(question) > 200 else question
            })

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

    def get_admin_stats(self) -> dict:
        """Get detailed stats for admin."""
        self._reset_if_new_day()

        # Total requests all time
        total_requests = sum(d.get('requests', 0) for d in self.daily_stats.values())

        # Days with activity
        days_with_data = len(self.daily_stats)
        avg_per_day = round(total_requests / days_with_data, 1) if days_with_data > 0 else 0

        # User stats
        user_stats = []
        for uid, udata in self.users.items():
            user_requests = udata.get('requests', [])
            requests_today = sum(1 for r in user_requests if r.get('date') == self.today)

            name_parts = []
            if udata.get('first_name'):
                name_parts.append(udata['first_name'])
            if udata.get('last_name'):
                name_parts.append(udata['last_name'])
            name = ' '.join(name_parts) or 'Без имени'

            user_stats.append({
                'user_id': uid,
                'name': name,
                'username': udata.get('username', ''),
                'total_requests': len(user_requests),
                'requests_today': requests_today,
                'last_seen': udata.get('last_seen', ''),
                'recent_questions': [r.get('question', '') for r in user_requests[-5:]]
            })

        # Sort by total requests descending
        user_stats.sort(key=lambda x: x['total_requests'], reverse=True)

        return {
            'date': self.today,
            'requests_today': self.requests_today,
            'limit': DAILY_REQUEST_LIMIT,
            'total_requests': total_requests,
            'days_tracked': days_with_data,
            'avg_per_day': avg_per_day,
            'total_users': len(self.users),
            'users': user_stats
        }
