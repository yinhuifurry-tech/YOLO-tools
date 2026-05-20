from datetime import datetime
from .base import BaseModule
from framework.events import Events


class HistoryManager(BaseModule):
    def __init__(self):
        super().__init__()
        self.records = []
        self.max_history = 100

    def on_start(self):
        self.max_history = self.config.get('history', 'max_history', 100)

    def add(self, image_path, classes_count):
        record = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'image': image_path if image_path else 'realtime',
            'objects': dict(classes_count),
            'total': sum(classes_count.values()),
        }
        self.records.append(record)
        if len(self.records) > self.max_history:
            self.records.pop(0)
        self.emit(Events.HISTORY_UPDATED, record=record)

    def clear(self):
        self.records.clear()
        self.emit(Events.HISTORY_UPDATED, record=None)

    def export(self, file_path):
        if file_path.endswith('.json'):
            import json
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(self.records, f, ensure_ascii=False, indent=2)
        else:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write("Detection History\n")
                f.write("=" * 50 + "\n\n")
                for record in self.records:
                    f.write(f"Time: {record['timestamp']}\n")
                    f.write(f"Image: {record['image']}\n")
                    f.write("Objects:\n")
                    for obj, count in record['objects'].items():
                        f.write(f"  - {obj}: {count}\n")
                    f.write(f"Total: {record['total']}\n")
                    f.write("-" * 50 + "\n\n")
        return True

    @property
    def recent(self, n=20):
        return list(reversed(self.records[-n:]))
