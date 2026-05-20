import os
import json
import time
import threading
from .base import BaseModule


class DetectionLogger(BaseModule):
    def __init__(self):
        super().__init__()
        self._log_path = 'detection_log.jsonl'
        self._interval = 10.0
        self._last_flush = 0.0
        self._buffer = []
        self._counts = {}
        self._lock = threading.Lock()
        self._enabled = True
        self._log_detections = False

    def on_start(self):
        self._log_path = self.config.get('logger', 'path', 'detection_log.jsonl')
        self._interval = float(self.config.get('logger', 'interval', '10'))
        self._log_detections = self.config.get('logger', 'log_detections', False)

    @property
    def interval(self):
        return self._interval

    @interval.setter
    def interval(self, v):
        self._interval = max(1.0, float(v))
        self.config.set('logger', 'interval', str(self._interval))

    @property
    def enabled(self):
        return self._enabled

    @enabled.setter
    def enabled(self, v):
        self._enabled = bool(v)

    @property
    def log_path(self):
        return self._log_path

    def log_detection(self, class_name, confidence, x1, y1, x2, y2, image_path='', source='detect'):
        if not self._enabled:
            return
        with self._lock:
            self._counts[class_name] = self._counts.get(class_name, 0) + 1
            if self._log_detections:
                self._buffer.append({
                    't': time.time(),
                    'class': class_name,
                    'conf': round(confidence, 4),
                    'bbox': [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
                    'img': image_path,
                    'src': source,
                })
            self._maybe_flush()

    def log_batch(self, detections, image_path='', source='detect'):
        if not self._enabled:
            return
        with self._lock:
            for d in detections:
                self._counts[d['class_name']] = self._counts.get(d['class_name'], 0) + 1
                if self._log_detections:
                    self._buffer.append({
                        't': time.time(),
                        'class': d['class_name'],
                        'conf': d['confidence'],
                        'bbox': [d['x1'], d['y1'], d['x2'], d['y2']],
                        'img': image_path,
                        'src': source,
                    })
            self._maybe_flush()

    def _maybe_flush(self):
        now = time.time()
        if now - self._last_flush >= self._interval:
            self._flush()

    def _flush(self):
        ts = time.strftime('%Y-%m-%d %H:%M:%S')
        summary = {
            'ts': ts,
            'type': 'summary',
            'elapsed_s': round(time.time() - self._last_flush, 1) if self._last_flush > 0 else 0,
            'classes': dict(self._counts),
            'total': sum(self._counts.values()),
        }
        with open(self._log_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(summary, ensure_ascii=False) + '\n')
            for entry in self._buffer:
                entry['ts'] = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(entry.pop('t')))
                entry['type'] = 'detection'
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')
        self._buffer.clear()
        self._counts.clear()
        self._last_flush = time.time()

    def force_flush(self):
        with self._lock:
            self._flush()

    def clear_log(self):
        with self._lock:
            if os.path.exists(self._log_path):
                open(self._log_path, 'w').close()
            self._buffer.clear()
            self._counts.clear()
            self._last_flush = time.time()

    @property
    def log_size_mb(self):
        if os.path.exists(self._log_path):
            return round(os.path.getsize(self._log_path) / (1024 * 1024), 2)
        return 0

    @property
    def log_lines(self):
        if os.path.exists(self._log_path):
            with open(self._log_path, 'r', encoding='utf-8') as f:
                return sum(1 for _ in f)
        return 0
