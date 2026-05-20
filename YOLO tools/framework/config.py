import os
import json


_defaults = {
    'model': {
        'path': None,
        'inference_mode': 'cpu',
        'input_size': 640,
        'conf_threshold': 0.35,
        'iou_threshold': 0.5,
    },
    'training': {
        'model': 'yolov8n.pt',
        'data': 'dataset.yaml',
        'epochs': 100,
        'batch': 8,
        'imgsz': 640,
        'device': '',
        'project': 'runs/train',
        'name': 'exp',
        'lr0': 0.0005,
        'lrf': 0.01,
        'momentum': 0.937,
        'weight_decay': 0.0005,
        'hsv_h': 0.015,
        'hsv_s': 0.7,
        'hsv_v': 0.4,
        'flipud': 0.0,
        'fliplr': 0.5,
        'mosaic': 1.0,
        'mixup': 0.0,
    },
    'inference': {
        'target_fps': 60,
        'skip_frames': 0,
        'use_frame_buffer': True,
        'max_queue_size': 2,
        'detection_interval': 2,
    },
    'label_studio': {
        'port': 5000,
    },
    'history': {
        'max_history': 100,
    },
    'logger': {
        'path': 'detection_log.jsonl',
        'interval': '10',
        'log_detections': False,
    },
}

CONFIG_FILE = 'config.json'


class Config:
    def __init__(self):
        self._data = {}
        for section, values in _defaults.items():
            self._data[section] = dict(values)
        self._load()

    def get(self, section, key, default=None):
        return self._data.get(section, {}).get(key, default)

    def set(self, section, key, value):
        if section not in self._data:
            self._data[section] = {}
        self._data[section][key] = value

    def section(self, section):
        return dict(self._data.get(section, {}))

    def dict(self):
        return {k: dict(v) for k, v in self._data.items()}

    def save(self):
        try:
            with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
                json.dump(self._data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"[Config] Failed to save config: {e}")

    def _load(self):
        if not os.path.exists(CONFIG_FILE):
            return
        try:
            with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
                saved = json.load(f)
            for section, values in saved.items():
                if section in self._data:
                    self._data[section].update(values)
        except Exception as e:
            print(f"[Config] Failed to load config: {e}")
