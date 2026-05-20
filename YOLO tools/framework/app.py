from .config import Config
from .events import EventBus, Events


class App:
    def __init__(self):
        self.config = Config()
        self.events = EventBus()
        self._modules = {}
        self._running = False

    def register(self, name, module):
        module._app = self
        module._name = name
        self._modules[name] = module
        module.on_register()

    def get_module(self, name):
        return self._modules.get(name)

    def start(self):
        self._running = True
        for name, module in self._modules.items():
            module.on_start()
        self.events.emit(Events.APP_START)

    def stop(self):
        self._running = False
        self.events.emit(Events.APP_STOP)
        for name, module in reversed(list(self._modules.items())):
            module.on_stop()
        self.config.save()

    @property
    def modules(self):
        return dict(self._modules)
