class BaseModule:
    def __init__(self):
        self._app = None
        self._name = None

    @property
    def app(self):
        return self._app

    @property
    def name(self):
        return self._name

    @property
    def config(self):
        return self._app.config if self._app else None

    @property
    def events(self):
        return self._app.events if self._app else None

    def on_register(self):
        pass

    def on_start(self):
        pass

    def on_stop(self):
        pass

    def get_module(self, name):
        if self._app:
            return self._app.get_module(name)
        return None

    def emit(self, event, **kwargs):
        if self._app:
            self._app.events.emit(event, **kwargs)

    def log(self, message):
        print(f"[{self._name}] {message}")
