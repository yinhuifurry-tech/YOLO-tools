import os
import sys
import tkinter as tk
from tkinter import ttk

from runtime_env import setup_style, get_dpi_scale

setup_style()
_DPI_SCALE = get_dpi_scale()
_FONT_FAMILY = 'Microsoft YaHei UI'


def create_app():
    app = App()
    app.register('model_loader', ModelLoader())
    app.register('inference', InferenceEngine())
    app.register('training', TrainingEngine())
    app.register('converter', ModelConverter())
    app.register('dataset', DatasetManager())
    app.register('label_studio', LabelStudioService())
    app.register('history', HistoryManager())
    app.register('logger', DetectionLogger())
    return app


class LauncherWindow:
    def __init__(self, app):
        self.root = tk.Tk()
        self.root.title(T('launcher.title'))
        s = _DPI_SCALE
        w, h = int(600 * s), int(480 * s)
        self.root.geometry(f"{w}x{h}")
        self.root.minsize(int(480 * s), int(380 * s))

        self.app = app
        self._tr_widgets = []

        self._create_widgets()
        self._open_windows = []
        on_change(self._refresh_i18n)

    def _t(self, key, default=''):
        return T(key, default)

    def _refresh_i18n(self):
        self.root.title(T('launcher.title'))
        for cb in self._tr_widgets:
            try:
                cb()
            except Exception:
                pass

    def _lbl(self, parent, key, **kw):
        lbl = ttk.Label(parent, text=self._t(key), **kw)
        self._tr_widgets.append(lambda k=key: lbl.config(text=self._t(k)))
        return lbl

    def _btn(self, parent, key, **kw):
        cmd = kw.pop('command', None)
        btn = ttk.Button(parent, text=self._t(key), **kw)
        self._tr_widgets.append(lambda k=key: btn.config(text=self._t(k)))
        if cmd:
            btn.config(command=cmd)
        return btn

    def _create_widgets(self):
        s = _DPI_SCALE
        pad = int(24 * s)
        frame = ttk.Frame(self.root, padding=pad)
        frame.pack(expand=True, fill=tk.BOTH)

        title_font = (os.environ.get('OP_YOLO_FONT', _FONT_FAMILY), max(22, int(24 * s)), 'bold')
        subtitle_font = (os.environ.get('OP_YOLO_FONT', _FONT_FAMILY), max(10, int(11 * s)))
        check_font = (os.environ.get('OP_YOLO_FONT', _FONT_FAMILY), max(9, int(10 * s)))

        lang_frame = ttk.Frame(frame)
        lang_frame.pack(anchor=tk.NE, pady=(0, int(12 * s)))
        self._lbl(lang_frame, 'launcher.language').pack(side=tk.LEFT, padx=(0, int(12 * s)))
        self.lang_zh_btn = ttk.Button(lang_frame, text="中文", command=lambda: self._switch_lang('zh'), width=max(6, int(7 * s)))
        self.lang_zh_btn.pack(side=tk.LEFT, padx=(0, int(6 * s)))
        self.lang_en_btn = ttk.Button(lang_frame, text="English", command=lambda: self._switch_lang('en'), width=max(7, int(8 * s)))
        self.lang_en_btn.pack(side=tk.LEFT)
        self._update_lang_buttons()

        ttk.Label(frame, text="YOLO AI Platform", font=title_font).pack(pady=(0, int(6 * s)))
        self._lbl(frame, 'launcher.subtitle', font=subtitle_font).pack(pady=(0, int(36 * s)))

        self._btn(frame, 'launcher.open_detect', command=self._open_detection, width=max(26, int(30 * s))).pack(pady=int(6 * s))
        self._btn(frame, 'launcher.open_train', command=self._open_training, width=max(26, int(30 * s))).pack(pady=int(6 * s))

        ttk.Separator(frame, orient='horizontal').pack(fill=tk.X, pady=int(24 * s))
        self._lbl(frame, 'launcher.quick_check').pack()
        self._lbl(frame, 'launcher.status_ok', font=check_font, foreground="green").pack(pady=int(6 * s))
        self._btn(frame, 'launcher.quit', command=self._on_quit, width=max(14, int(16 * s))).pack(pady=(int(18 * s), 0))

    def _update_lang_buttons(self):
        lang = current()
        self.lang_zh_btn.config(state="disabled" if lang == 'zh' else "normal")
        self.lang_en_btn.config(state="disabled" if lang == 'en' else "normal")

    def _switch_lang(self, lang):
        set_lang(lang)
        self._update_lang_buttons()

    def _open_detection(self):
        win = DetectionWindow(self.app)
        self._open_windows.append(win)

    def _open_training(self):
        win = TrainingWindow(self.app)
        self._open_windows.append(win)

    def _on_quit(self):
        self.app.stop()
        for win in self._open_windows:
            try:
                if win.root.winfo_exists():
                    win.root.destroy()
            except Exception:
                pass
        off_change(self._refresh_i18n)
        self.root.destroy()

    def run(self):
        self.root.mainloop()


def main():
    import argparse
    parser = argparse.ArgumentParser(description="YOLO AI Platform")
    parser.add_argument('--detection', action='store_true', help='Launch detection GUI directly')
    parser.add_argument('--training', action='store_true', help='Launch training GUI directly')
    parser.add_argument('--lang', type=str, default='zh', choices=['zh', 'en'],
                        help='Default language (zh/en)')
    args = parser.parse_args()

    set_lang(args.lang)
    app = create_app()
    app.start()

    if args.detection and not args.training:
        root = tk.Tk()
        root.withdraw()
        win = DetectionWindow(app)
        root.mainloop()
    elif args.training and not args.detection:
        root = tk.Tk()
        root.withdraw()
        win = TrainingWindow(app)
        root.mainloop()
    else:
        launcher = LauncherWindow(app)
        launcher.run()

    app.stop()


if __name__ == "__main__":
    main()
