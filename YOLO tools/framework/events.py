from enum import Enum, auto
import threading


class Events(Enum):
    APP_START = auto()
    APP_STOP = auto()
    MODEL_LOADED = auto()
    MODEL_UNLOADED = auto()
    DETECTION_STARTED = auto()
    DETECTION_COMPLETED = auto()
    DETECTION_FRAME = auto()
    TRAINING_STARTED = auto()
    TRAINING_COMPLETED = auto()
    TRAINING_PROGRESS = auto()
    CONVERSION_STARTED = auto()
    CONVERSION_COMPLETED = auto()
    LS_SERVICE_STARTED = auto()
    LS_SERVICE_STOPPED = auto()
    CAMERA_STARTED = auto()
    CAMERA_STOPPED = auto()
    VIDEO_PLAY = auto()
    VIDEO_PAUSE = auto()
    VIDEO_SEEK = auto()
    HISTORY_UPDATED = auto()
    STATUS_CHANGED = auto()
    ERROR = auto()


class EventBus:
    def __init__(self):
        self._listeners = {}
        self._lock = threading.RLock()

    def on(self, event, callback):
        with self._lock:
            if event not in self._listeners:
                self._listeners[event] = []
            self._listeners[event].append(callback)

    def off(self, event, callback):
        with self._lock:
            if event in self._listeners:
                try:
                    self._listeners[event].remove(callback)
                except ValueError:
                    pass

    def emit(self, event, **kwargs):
        with self._lock:
            listeners = list(self._listeners.get(event, []))
        for callback in listeners:
            try:
                callback(**kwargs)
            except Exception as e:
                import traceback
                print(f"[EventBus] Error in {event.name}: {e}")
                traceback.print_exc()
