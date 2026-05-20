import os
import json
import sys
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
from PIL import Image, ImageTk
Image.MAX_IMAGE_PIXELS = None
from framework.events import Events
from framework.i18n import T, on_change, off_change

_SCALE = 1.0
try:
    if 'main' in sys.modules:
        _SCALE = sys.modules['main'].get_dpi_scale()
except Exception:
    pass

_IMG_MAX_W = int(860 * _SCALE)
_IMG_MAX_H = int(520 * _SCALE)


class DetectionWindow:
    def __init__(self, app):
        self.app = app
        self.root = tk.Toplevel()
        self.root.title(T('detect.title'))
        w, h = int(1400 * _SCALE), int(920 * _SCALE)
        self.root.geometry(f"{w}x{h}")
        self.root.minsize(int(1100 * _SCALE), int(700 * _SCALE))

        self.model_loader = app.get_module('model_loader')
        self.inference = app.get_module('inference')
        self.history_mgr = app.get_module('history')
        self.logger = app.get_module('logger')

        self.image_paths = []
        self.current_image_index = 0
        self.detection_results = {}
        self._orig_image_cache = {}
        self._tr_widgets = []
        self._bound_events = []
        self._class_filter_vars = {}
        self._class_filter_cbs = []

        self._create_widgets()
        self._bind_events()
        self._bind_keys()

        on_change(self._refresh_i18n)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _t(self, key, default=''):
        return T(key, default)

    def _refresh_i18n(self):
        self.root.title(T('detect.title'))
        for cb in self._tr_widgets:
            try:
                cb()
            except Exception:
                pass
        self._update_class_filter_panel()

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

    def _frm(self, parent, key, **kw):
        frm = ttk.LabelFrame(parent, text=self._t(key), **kw)
        self._tr_widgets.append(lambda k=key: frm.configure(text=self._t(k)))
        return frm

    def _bind_events(self):
        pairs = [
            (Events.MODEL_LOADED, self._on_model_loaded),
            (Events.MODEL_UNLOADED, self._on_model_unloaded),
            (Events.HISTORY_UPDATED, self._on_history_updated),
            (Events.CAMERA_STARTED, self._on_camera_started),
            (Events.CAMERA_STOPPED, self._on_camera_stopped),
            (Events.LS_SERVICE_STARTED, self._on_ls_started),
            (Events.LS_SERVICE_STOPPED, self._on_ls_stopped),
        ]
        for ev, cb in pairs:
            self.app.events.on(ev, cb)
            self._bound_events.append((ev, cb))

    def _unbind_events(self):
        for ev, cb in self._bound_events:
            self.app.events.off(ev, cb)
        self._bound_events.clear()

    def _bind_keys(self):
        self.root.bind('<space>', lambda e: self._toggle_play())
        self.root.bind('<Left>', lambda e: self._show_prev())
        self.root.bind('<Right>', lambda e: self._show_next())
        self.root.bind('<Control-s>', lambda e: self._save_all())
        self.root.bind('<Key-s>', lambda e: self._save_current())
        self.root.bind('<Control-e>', lambda e: self._export_json())
        self.root.bind('<Key-c>', lambda e: self._toggle_camera())
        self.root.bind('<Escape>', lambda e: self._clear_results())
        self.root.bind('<F5>', lambda e: self._refresh_cameras())
        self.root.bind('<Key-p>', lambda e: self._snapshot())

    # ==================== LAYOUT ====================
    def _create_widgets(self):
        main_frame = ttk.Frame(self.root, padding="6")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # ---- Row 0: Toolbar ----
        toolbar = ttk.Frame(main_frame)
        toolbar.pack(fill=tk.X, pady=(0, 4))

        self._btn(toolbar, 'detect.export_json', command=self._export_json, width=11).pack(side=tk.LEFT, padx=(0, 3))
        self._btn(toolbar, 'detect.export_csv', command=self._export_csv, width=11).pack(side=tk.LEFT, padx=(0, 3))
        self._btn(toolbar, 'detect.snapshot', command=self._snapshot, width=11).pack(side=tk.LEFT, padx=(0, 3))
        ttk.Separator(toolbar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=8)

        self._lbl(toolbar, 'detect.confidence').pack(side=tk.LEFT, padx=(0, 3))
        self.conf_var = tk.StringVar(value='0.35')
        self.conf_slider = ttk.Scale(toolbar, from_=0.05, to=1.0, variable=self.conf_var,
                                      orient=tk.HORIZONTAL, length=120, command=self._on_conf_changed)
        self.conf_slider.pack(side=tk.LEFT, padx=(0, 3))
        self.conf_label = ttk.Label(toolbar, text="0.35", width=5)
        self.conf_label.pack(side=tk.LEFT, padx=(0, 12))

        self._lbl(toolbar, 'detect.iou').pack(side=tk.LEFT, padx=(0, 3))
        self.iou_var = tk.StringVar(value='0.5')
        ttk.Scale(toolbar, from_=0.1, to=1.0, variable=self.iou_var,
                   orient=tk.HORIZONTAL, length=80, command=self._on_iou_changed).pack(side=tk.LEFT, padx=(0, 3))
        self.iou_label = ttk.Label(toolbar, text="0.5", width=5)
        self.iou_label.pack(side=tk.LEFT, padx=(0, 12))

        ttk.Separator(toolbar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=8)
        self.fps_label = ttk.Label(toolbar, text="FPS: --", font=("Consolas", 10, "bold"), foreground="gray")
        self.fps_label.pack(side=tk.LEFT, padx=(0, 12))

        self.det_count_label = ttk.Label(toolbar, text="Det: 0", font=("Consolas", 10), foreground="gray")
        self.det_count_label.pack(side=tk.LEFT, padx=(0, 12))

        self._lbl(toolbar, 'detect.shortcuts_hint', font=("Arial", 8), foreground="gray").pack(side=tk.RIGHT)

        # ---- Row 1: Model & File ----
        top_area = ttk.Frame(main_frame)
        top_area.pack(fill=tk.X, pady=(0, 4))

        model_frame = self._frm(top_area, 'detect.model_group', padding="4")
        model_frame.pack(fill=tk.X, side=tk.LEFT, expand=True, padx=(0, 3))
        model_frame.columnconfigure(1, weight=1)

        self.model_path_var = tk.StringVar()
        self._lbl(model_frame, 'detect.model_label').grid(row=0, column=0, sticky=tk.W, padx=(0, 3))
        ttk.Entry(model_frame, textvariable=self.model_path_var, state="readonly", width=35).grid(
            row=0, column=1, sticky=(tk.W, tk.E), padx=(0, 3))
        self._btn(model_frame, 'detect.browse', command=self._select_model, width=7).grid(row=0, column=2, padx=(0, 2))
        self._btn(model_frame, 'detect.unload', command=self._unload_model, width=7).grid(row=0, column=3)

        file_frame = self._frm(top_area, 'detect.file_group', padding="4")
        file_frame.pack(fill=tk.X, side=tk.LEFT, expand=True, padx=(3, 0))
        file_frame.columnconfigure(1, weight=1)

        self.image_path_var = tk.StringVar()
        self._lbl(file_frame, 'detect.images').grid(row=0, column=0, sticky=tk.W, padx=(0, 3))
        ttk.Entry(file_frame, textvariable=self.image_path_var, state="readonly", width=30).grid(
            row=0, column=1, sticky=(tk.W, tk.E), padx=(0, 3))
        self._btn(file_frame, 'detect.select_images', command=self._select_images, width=9).grid(row=0, column=2)

        self.video_path_var = tk.StringVar()
        self._lbl(file_frame, 'detect.video').grid(row=1, column=0, sticky=tk.W, padx=(0, 3), pady=(3, 0))
        ttk.Entry(file_frame, textvariable=self.video_path_var, state="readonly", width=30).grid(
            row=1, column=1, sticky=(tk.W, tk.E), padx=(0, 3), pady=(3, 0))
        self._btn(file_frame, 'detect.select_video', command=self._select_video, width=9).grid(row=1, column=2, pady=(3, 0))

        # ---- Row 2: Left (Controls) | Center (Image) | Right (Table) ----
        body = ttk.Frame(main_frame)
        body.pack(fill=tk.BOTH, expand=True)
        body.columnconfigure(1, weight=1)
        body.rowconfigure(0, weight=1)

        # LEFT COLUMN: Controls
        left_col = ttk.Frame(body, width=240)
        left_col.grid(row=0, column=0, sticky=(tk.N, tk.S), padx=(0, 3))
        left_col.pack_propagate(False)

        # Detection group
        det_grp = self._frm(left_col, 'detect.det_group', padding="4")
        det_grp.pack(fill=tk.X, pady=(0, 3))

        self.detect_btn = self._btn(det_grp, 'detect.batch_detect', command=self._start_batch_detection, state="disabled")
        self.detect_btn.pack(fill=tk.X, pady=1, padx=1)

        self.video_gen_btn = self._btn(det_grp, 'detect.gen_video',
                                        command=self._start_video_generation, state="disabled")
        self.video_gen_btn.pack(fill=tk.X, pady=1, padx=1)

        # Camera group
        cam_grp = self._frm(left_col, 'detect.cam_group', padding="4")
        cam_grp.pack(fill=tk.X, pady=(0, 3))

        cam_row1 = ttk.Frame(cam_grp)
        cam_row1.pack(fill=tk.X)
        self._lbl(cam_row1, 'detect.cam_index').pack(side=tk.LEFT, padx=(0, 2))
        self.camera_idx_var = tk.StringVar(value="0")
        self.camera_idx_combo = ttk.Combobox(cam_row1, textvariable=self.camera_idx_var,
                                              values=["0", "1", "2", "3"], width=4)
        self.camera_idx_combo.pack(side=tk.LEFT, padx=(0, 2))
        self._btn(cam_row1, 'detect.cam_refresh', command=self._refresh_cameras, width=6).pack(side=tk.LEFT, padx=(0, 2))
        self.camera_btn = self._btn(cam_grp, 'detect.cam_start', command=self._toggle_camera, state="disabled")
        self.camera_btn.pack(fill=tk.X, pady=1, padx=1)
        self.camera_status_label = ttk.Label(cam_grp, text=self._t('detect.cam_disconnected'), foreground="gray")
        self.camera_status_label.pack(anchor=tk.W)
        self._tr_widgets.append(lambda: self._update_camera_status_label())

        # LS group
        ls_grp = self._frm(left_col, 'detect.ls_group', padding="4")
        ls_grp.pack(fill=tk.X, pady=(0, 3))

        self.ls_btn = self._btn(ls_grp, 'detect.ls_start', command=self._toggle_ls_service, state="disabled")
        self.ls_btn.pack(fill=tk.X, pady=1, padx=1)
        self.ls_status_label = ttk.Label(ls_grp, text=self._t('detect.ls_stopped'), foreground="gray")
        self.ls_status_label.pack(anchor=tk.W)
        self._tr_widgets.append(lambda: self._update_ls_status_label())

        # Playback group
        vid_grp = self._frm(left_col, 'detect.playback', padding="4")
        vid_grp.pack(fill=tk.X, pady=(0, 3))

        vid_btns = ttk.Frame(vid_grp)
        vid_btns.pack(fill=tk.X)
        self.play_btn = self._btn(vid_btns, 'detect.play', command=self._toggle_play, state="disabled")
        self.play_btn.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 2))
        self.pause_btn = self._btn(vid_btns, 'detect.pause', command=self._pause_video, state="disabled")
        self.pause_btn.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=(2, 0))

        self.video_progress = ttk.Scale(vid_grp, from_=0, to=100, orient=tk.HORIZONTAL,
                                         state="disabled", command=self._on_video_seek)
        self.video_progress.pack(fill=tk.X, pady=(3, 1))
        self.time_label = ttk.Label(vid_grp, text="00:00 / 00:00")
        self.time_label.pack()

        # Navigation
        nav_grp = self._frm(left_col, 'detect.nav_group', padding="4")
        nav_grp.pack(fill=tk.X, pady=(0, 3))

        nav_btns = ttk.Frame(nav_grp)
        nav_btns.pack(fill=tk.X)
        self.prev_btn = self._btn(nav_btns, 'detect.prev', command=self._show_prev, state="disabled")
        self.prev_btn.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 2))
        self.next_btn = self._btn(nav_btns, 'detect.next', command=self._show_next, state="disabled")
        self.next_btn.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=(2, 0))
        self.index_label = ttk.Label(nav_grp, text="0 / 0")
        self.index_label.pack(pady=(3, 0))

        # Output
        io_grp = self._frm(left_col, 'detect.output_group', padding="4")
        io_grp.pack(fill=tk.X, pady=(0, 3))

        self.save_btn = self._btn(io_grp, 'detect.save_current', command=self._save_current, state="disabled")
        self.save_btn.pack(fill=tk.X, pady=1, padx=1)
        self.save_all_btn = self._btn(io_grp, 'detect.save_all', command=self._save_all, state="disabled")
        self.save_all_btn.pack(fill=tk.X, pady=1, padx=1)
        self.clear_btn = self._btn(io_grp, 'detect.clear_all', command=self._clear_results)
        self.clear_btn.pack(fill=tk.X, pady=1, padx=1)

        # Logger
        log_grp = self._frm(left_col, 'detect.log_group', padding="4")
        log_grp.pack(fill=tk.X, pady=(0, 3))

        log_row1 = ttk.Frame(log_grp)
        log_row1.pack(fill=tk.X)
        self._lbl(log_row1, 'detect.log_interval').pack(side=tk.LEFT, padx=(0, 3))
        self.log_interval_var = tk.StringVar(value='10')
        ttk.Spinbox(log_row1, textvariable=self.log_interval_var, from_=1, to=300, increment=5, width=5,
                     command=self._on_log_interval_changed).pack(side=tk.LEFT, padx=(0, 5))
        self._lbl(log_row1, 'detect.log_unit').pack(side=tk.LEFT)

        self._btn(log_grp, 'detect.log_flush', command=self._flush_log, width=10).pack(fill=tk.X, pady=1, padx=1)

        self.log_info_label = ttk.Label(log_grp, text=self._t('detect.log_empty'), foreground="gray", font=("Arial", 8))
        self.log_info_label.pack(anchor=tk.W, pady=(2, 0))
        self._tr_widgets.append(lambda: self._update_log_label())

        # CENTER: Image + History
        center_col = ttk.Frame(body)
        center_col.grid(row=0, column=1, sticky=(tk.N, tk.S, tk.E, tk.W), padx=3)
        center_col.columnconfigure(0, weight=1)
        center_col.rowconfigure(0, weight=3)
        center_col.rowconfigure(1, weight=1)

        self.image_label = ttk.Label(center_col, text=self._t('detect.placeholder'),
                                      background="lightgray", anchor=tk.CENTER)
        self.image_label.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        hist_frame = self._frm(center_col, 'detect.hist_group', padding="4")
        hist_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(3, 0))
        hist_frame.columnconfigure(0, weight=1)
        hist_frame.rowconfigure(0, weight=1)

        self.history_text = tk.Text(hist_frame, height=4, width=30, state='disabled', font=("Consolas", 8))
        h_scroll = ttk.Scrollbar(hist_frame, orient=tk.VERTICAL, command=self.history_text.yview)
        self.history_text.configure(yscrollcommand=h_scroll.set)
        self.history_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        h_scroll.grid(row=0, column=1, sticky=(tk.N, tk.S))

        hist_btns = ttk.Frame(hist_frame)
        hist_btns.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(3, 0))
        self._btn(hist_btns, 'detect.hist_clear', command=self._clear_history).pack(side=tk.LEFT, padx=(0, 3))
        self._btn(hist_btns, 'detect.hist_export', command=self._export_history).pack(side=tk.LEFT)

        # RIGHT: Detection Table + Class Filter
        right_col = ttk.Frame(body, width=320)
        right_col.grid(row=0, column=2, sticky=(tk.N, tk.S, tk.E, tk.W), padx=(3, 0))
        right_col.pack_propagate(False)
        right_col.rowconfigure(0, weight=2)
        right_col.rowconfigure(1, weight=1)

        # Detection table
        table_frame = self._frm(right_col, 'detect.result_group', padding="4")
        table_frame.grid(row=0, column=0, sticky=(tk.N, tk.S, tk.E, tk.W))
        table_frame.columnconfigure(0, weight=1)
        table_frame.rowconfigure(0, weight=1)

        columns = ('#', 'class', 'conf', 'bbox')
        self.det_tree = ttk.Treeview(table_frame, columns=columns, show='headings',
                                      height=12, selectmode='browse')
        self.det_tree.heading('#', text='#', command=lambda: self._sort_tree('#', False))
        self.det_tree.heading('class', text=T('detect.tbl_class'),
                               command=lambda: self._sort_tree('class', False))
        self.det_tree.heading('conf', text=T('detect.tbl_conf'),
                               command=lambda: self._sort_tree('conf', False))
        self.det_tree.heading('bbox', text=T('detect.tbl_bbox'),
                               command=lambda: self._sort_tree('bbox', False))
        self.det_tree.column('#', width=30, anchor=tk.CENTER, stretch=False)
        self.det_tree.column('class', width=90, anchor=tk.W)
        self.det_tree.column('conf', width=55, anchor=tk.CENTER)
        self.det_tree.column('bbox', width=130, anchor=tk.CENTER)

        tree_scroll = ttk.Scrollbar(table_frame, orient=tk.VERTICAL, command=self.det_tree.yview)
        self.det_tree.configure(yscrollcommand=tree_scroll.set)
        self.det_tree.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        tree_scroll.grid(row=0, column=1, sticky=(tk.N, tk.S))

        # Click to highlight box on image
        self.det_tree.bind('<<TreeviewSelect>>', self._on_tree_select)
        self.det_tree.bind('<Double-1>', lambda e: self._on_tree_double_click())
        self._tr_widgets.append(self._update_tree_headers)

        # Class filter
        filter_frame = self._frm(right_col, 'detect.class_filter', padding="4")
        filter_frame.grid(row=1, column=0, sticky=(tk.N, tk.S, tk.E, tk.W), pady=(3, 0))
        filter_frame.columnconfigure(0, weight=1)
        filter_frame.rowconfigure(1, weight=1)

        filter_toolbar = ttk.Frame(filter_frame)
        filter_toolbar.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 2))
        self._btn(filter_toolbar, 'detect.select_all', command=lambda: self._toggle_all_classes(True), width=7).pack(
            side=tk.LEFT, padx=(0, 3))
        self._btn(filter_toolbar, 'detect.deselect_all', command=lambda: self._toggle_all_classes(False), width=7).pack(
            side=tk.LEFT)

        self.filter_canvas = tk.Canvas(filter_frame, highlightthickness=0)
        filter_scroll = ttk.Scrollbar(filter_frame, orient=tk.VERTICAL, command=self.filter_canvas.yview)
        self.filter_inner = ttk.Frame(self.filter_canvas)
        self.filter_inner.bind("<Configure>",
                                lambda e: self.filter_canvas.configure(scrollregion=self.filter_canvas.bbox("all")))
        self.filter_canvas.create_window((0, 0), window=self.filter_inner, anchor=tk.NW)
        self.filter_canvas.configure(yscrollcommand=filter_scroll.set)
        self.filter_canvas.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        filter_scroll.grid(row=1, column=1, sticky=(tk.N, tk.S))

        # ---- Bottom: Status ----
        status_bar = ttk.Frame(main_frame)
        status_bar.pack(fill=tk.X, pady=(4, 0))

        self.progress = ttk.Progressbar(status_bar, mode='indeterminate', length=150)
        self.progress.pack(side=tk.LEFT, padx=(0, 5))

        self.status_var = tk.StringVar(value=self._t('detect.status_ready'))
        ttk.Label(status_bar, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W).pack(
            side=tk.LEFT, fill=tk.X, expand=True)

        self._lbl(status_bar, 'detect.class_count', font=("Arial", 9), foreground="gray").pack(side=tk.RIGHT, padx=(5, 0))

    def _update_tree_headers(self):
        self.det_tree.heading('class', text=T('detect.tbl_class'))
        self.det_tree.heading('conf', text=T('detect.tbl_conf'))
        self.det_tree.heading('bbox', text=T('detect.tbl_bbox'))

    # ==================== MODEL ====================
    def _update_model_buttons(self, enabled):
        state = "normal" if enabled else "disabled"
        for btn in [self.detect_btn, self.video_gen_btn, self.camera_btn, self.ls_btn]:
            btn.config(state=state)

    def _select_model(self):
        path = filedialog.askopenfilename(
            title=T('detect.model_label'),
            filetypes=[("YOLO Models", "*.pt *.onnx"), ("PyTorch", "*.pt"), ("ONNX", "*.onnx"), ("All", "*.*")]
        )
        if not path:
            return
        try:
            self.model_loader.load(path)
            self.model_path_var.set(path)
            self._update_model_buttons(True)
            self.status_var.set(f"{T('detect.status_loaded')} {os.path.basename(path)}")
            self._rebuild_class_filter()
        except Exception as e:
            messagebox.showerror(T('dlg.error'), f"{T('dlg.load_failed')}:\n{e}")

    def _unload_model(self):
        self.model_loader.unload()
        self.model_path_var.set("")
        self._update_model_buttons(False)
        self.status_var.set(T('detect.status_unloaded'))
        self._clear_class_filter()

    def _on_model_loaded(self, **kwargs):
        self.root.after(0, lambda: self.status_var.set(
            f"{T('detect.status_loaded')} {os.path.basename(kwargs.get('path', ''))}"))
        self.root.after(0, self._rebuild_class_filter)

    def _on_model_unloaded(self, **kwargs):
        self.root.after(0, lambda: self._update_model_buttons(False))

    # ==================== CLASS FILTER ====================
    def _rebuild_class_filter(self):
        self._clear_class_filter()
        if not self.model_loader.is_loaded:
            return
        for cid, cname in sorted(self.model_loader.names.items()):
            var = tk.BooleanVar(value=True)
            self._class_filter_vars[cid] = var
            row = ttk.Frame(self.filter_inner)
            row.pack(fill=tk.X, anchor=tk.W)
            cb = ttk.Checkbutton(row, variable=var, command=self._on_class_filter_changed)
            cb.pack(side=tk.LEFT)
            lbl = ttk.Label(row, text=f"{cname} (id:{cid})", font=("Arial", 9))
            lbl.pack(side=tk.LEFT, padx=(2, 0))
            self._class_filter_cbs.append((cid, cb, lbl, var))
        self._apply_class_filter()

    def _clear_class_filter(self):
        self._class_filter_vars.clear()
        self._class_filter_cbs.clear()
        for child in self.filter_inner.winfo_children():
            child.destroy()

    def _update_class_filter_panel(self):
        for cid, cb, lbl, var in self._class_filter_cbs:
            cname = self.model_loader.names.get(cid, f'class_{cid}')
            lbl.config(text=f"{cname} (id:{cid})")

    def _on_class_filter_changed(self):
        self._apply_class_filter()

    def _apply_class_filter(self):
        enabled = set()
        for cid, var in self._class_filter_vars.items():
            if var.get():
                enabled.add(cid)
        if len(enabled) == len(self._class_filter_vars) and len(enabled) > 0:
            self.inference.set_class_filter(None)
        elif len(enabled) == 0:
            self.inference.set_class_filter(set())
        else:
            self.inference.set_class_filter(enabled)

    def _toggle_all_classes(self, state):
        for cid, var in self._class_filter_vars.items():
            var.set(state)
        self._apply_class_filter()

    # ==================== IMAGES ====================
    def _select_images(self):
        paths = filedialog.askopenfilenames(
            title=T('detect.select_images'),
            filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp *.tiff *.tif"), ("All", "*.*")]
        )
        if paths:
            self.image_paths = list(paths)
            self.current_image_index = 0
            self.detection_results = {}
            self._orig_image_cache.clear()
            self.image_path_var.set(f"{len(self.image_paths)} images selected")
            self._update_nav()
            if self.image_paths:
                self._display_image(self.image_paths[0])
            self.detect_btn.config(state="normal" if self.model_loader.is_loaded else "disabled")

    def _display_image(self, image_path):
        try:
            pil_image = Image.open(image_path)
            self._current_pil_image = pil_image
            pil_image = self._fit_image(pil_image, _IMG_MAX_W, _IMG_MAX_H)
            self._photo = ImageTk.PhotoImage(pil_image)
            self.image_label.configure(image=self._photo)
        except Exception as e:
            self.image_label.configure(image="", text=f"{T('dlg.img_load_failed')}: {e}")

    def _fit_image(self, image, max_w, max_h):
        w, h = image.size
        ratio = min(max_w / w, max_h / h, 1.0)
        return image.resize((int(w * ratio), int(h * ratio)), Image.Resampling.LANCZOS)

    def _update_nav(self):
        self.prev_btn.config(state="normal" if self.current_image_index > 0 else "disabled")
        has_next = self.current_image_index < len(self.image_paths) - 1
        self.next_btn.config(state="normal" if has_next else "disabled")
        total = len(self.image_paths)
        self.index_label.config(text=f"{self.current_image_index + 1} / {total}" if total else "0 / 0")

    def _show_next(self):
        if self.current_image_index < len(self.image_paths) - 1:
            self.current_image_index += 1
            self._show_current()

    def _show_prev(self):
        if self.current_image_index > 0:
            self.current_image_index -= 1
            self._show_current()

    def _show_current(self):
        path = self.image_paths[self.current_image_index]
        self._update_nav()
        if path in self.detection_results:
            self._display_with_results(path)
        else:
            self._display_image(path)
            self._clear_table()
            self.save_btn.config(state="disabled")

    def _display_with_results(self, path):
        results = self.detection_results[path]
        orig = self._orig_image_cache.get(path)
        if orig is None:
            orig = Image.open(path)
            self._orig_image_cache[path] = orig
        result_img = self.inference.draw_detections(orig, results)
        self._display_result_image(result_img)
        detections = self.inference.extract_detections(results)
        self._populate_table(detections)
        self._update_det_count_label(len(detections))
        self.save_btn.config(state="normal")

    # ==================== BATCH DETECTION ====================
    def _start_batch_detection(self):
        if not self.model_loader.is_loaded:
            messagebox.showwarning(T('dlg.warning'), T('dlg.no_model'))
            return
        if not self.image_paths:
            messagebox.showwarning(T('dlg.warning'), T('dlg.no_model_image'))
            return
        self._apply_confidence()
        self.progress.start()
        self.detect_btn.config(state="disabled")
        self.save_btn.config(state="disabled")
        self.save_all_btn.config(state="disabled")
        threading.Thread(target=self._run_batch_detection, daemon=True).start()

    def _apply_confidence(self):
        conf = float(self.conf_var.get())
        iou = float(self.iou_var.get())
        self.model_loader.config.set('model', 'conf_threshold', conf)
        self.model_loader.config.set('model', 'iou_threshold', iou)
        self.conf_label.config(text=f"{conf:.2f}")
        self.iou_label.config(text=f"{iou:.2f}")

    def _on_conf_changed(self, v):
        self.conf_label.config(text=f"{float(v):.2f}")
        self._apply_confidence()

    def _on_iou_changed(self, v):
        self.iou_label.config(text=f"{float(v):.2f}")
        self._apply_confidence()

    def _run_batch_detection(self):
        try:
            total = len(self.image_paths)
            errors = []
            for idx, path in enumerate(self.image_paths):
                self.root.after(0, lambda i=idx: self.status_var.set(
                    f"{T('detect.status_detecting')} {i+1}/{total}: {os.path.basename(self.image_paths[i])}"))
                if not os.path.exists(path):
                    msg = f"File missing: {path}"
                    errors.append(msg)
                    continue
                if not os.path.isfile(path):
                    msg = f"Not a file: {path}"
                    errors.append(msg)
                    continue
                fname = os.path.basename(path).lower()
                if fname in ('image.png', 'results.png', 'labels.jpg', 'confusion_matrix.png',
                              'pr_curve.png', 'f1_curve.png', 'p_curve.png', 'r_curve.png',
                              'labels_correlogram.jpg', 'train_batch.jpg', 'val_batch_pred.jpg',
                              'val_batch_labels.jpg'):
                    continue
                try:
                    results = self.model_loader.predict(path)
                    self.detection_results[path] = results
                    if self.logger:
                        dets = self.inference.extract_detections(results)
                        self.logger.log_batch(dets, path)
                except Exception as e:
                    msg = f"{os.path.basename(path)}: {e}"
                    errors.append(msg)
                    continue
                if self.current_image_index == idx:
                    self.root.after(0, lambda p=path, r=results: self._display_with_results(p))
            self.root.after(0, lambda: self.save_all_btn.config(state="normal"))
            if errors:
                self.root.after(0, lambda: messagebox.showwarning(
                    T('dlg.warning'),
                    f"{len(errors)}/{total} image(s) failed:\n" + "\n".join(errors[:5])))
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.root.after(0, lambda: messagebox.showerror(
                T('dlg.error'), f"Batch detection crashed:\n{e}"))
        finally:
            self.root.after(0, self._finish_batch)

    def _finish_batch(self):
        self.progress.stop()
        self.detect_btn.config(state="normal")
        self.status_var.set(T('detect.status_done'))

    def _display_result_image(self, image):
        self._current_pil_image = image
        image = self._fit_image(image, _IMG_MAX_W, _IMG_MAX_H)
        self._result_photo = ImageTk.PhotoImage(image)
        self.image_label.configure(image=self._result_photo)

    # ==================== DETECTION TABLE ====================
    def _populate_table(self, detections):
        self._clear_table()
        for d in detections:
            bbox_str = f"{d['x1']:.0f},{d['y1']:.0f} ~ {d['x2']:.0f},{d['y2']:.0f}"
            self.det_tree.insert('', tk.END,
                                  values=(d['id'], d['class_name'], f"{d['confidence']:.2%}", bbox_str),
                                  tags=('det_row',))
        self._update_class_summary(detections)

    def _clear_table(self):
        for item in self.det_tree.get_children():
            self.det_tree.delete(item)

    def _sort_tree(self, col, reverse):
        data = [(self.det_tree.set(item, col), item) for item in self.det_tree.get_children('')]
        try:
            data.sort(key=lambda x: float(x[0].rstrip('%')), reverse=reverse)
        except ValueError:
            data.sort(reverse=reverse)
        for idx, (_, item) in enumerate(data):
            self.det_tree.move(item, '', idx)
        self.det_tree.heading(col, command=lambda: self._sort_tree(col, not reverse))

    def _on_tree_select(self, event):
        pass

    def _on_tree_double_click(self):
        sel = self.det_tree.selection()
        if sel:
            item = self.det_tree.item(sel[0])
            values = item['values']
            if values:
                self.status_var.set(f"Selected: {values[1]} ({values[2]}) - {values[3]}")

    def _update_det_count_label(self, count):
        self.det_count_label.config(text=f"Det: {count}")

    def _update_class_summary(self, detections):
        summary = self.inference.class_summary(detections)
        total = len(detections)
        classes_str = ', '.join([f"{cn}:{v['count']}" for cn, v in sorted(summary.items())])
        self.status_var.set(f"{T('detect.total_objects')} {total} | {classes_str}")

        if self.current_image_index < len(self.image_paths):
            p = self.image_paths[self.current_image_index]
            self.history_mgr.add(os.path.basename(p),
                                  {cn: v['count'] for cn, v in summary.items()})

    # ==================== VIDEO ====================
    def _select_video(self):
        path = filedialog.askopenfilename(
            title=T('detect.select_video'),
            filetypes=[("Videos", "*.mp4 *.avi *.mov *.mkv *.wmv *.flv *.webm"), ("All", "*.*")]
        )
        if path:
            self.inference.load_video(path)
            self.video_path_var.set(os.path.basename(path))
            self.play_btn.config(state="normal")
            self.pause_btn.config(state="normal")
            self.video_progress.config(state="normal")
            self.video_gen_btn.config(state="normal" if self.model_loader.is_loaded else "disabled")
            fps = self.inference.video_cap.get(5)
            dur = self.inference.total_frames / fps if fps > 0 else 0
            self.status_var.set(f"Video: {self.inference.total_frames} frames, {dur:.1f}s")

    def _toggle_play(self):
        if not self.inference.playing:
            self._apply_confidence()
            self.inference.play_video(on_frame_callback=self._on_video_frame_ext)
            self.play_btn.config(text=T('detect.pause'))
            self.video_progress.config(state="disabled")
        else:
            self._pause_video()

    def _on_video_frame_ext(self, pil_img, cf, tf, fps_src, current_fps, detections_data):
        self.root.after(0, lambda: self._display_result_image(pil_img))
        prog = int((cf / tf) * 100) if tf > 0 else 0
        self.root.after(0, lambda p=prog: self.video_progress.set(p))
        ct = cf / fps_src if fps_src > 0 else 0
        tt = tf / fps_src if fps_src > 0 else 0
        ts = f"{int(ct // 60):02d}:{int(ct % 60):02d} / {int(tt // 60):02d}:{int(tt % 60):02d}"
        self.root.after(0, lambda t=ts: self.time_label.config(text=t))
        self.root.after(0, lambda f=current_fps: self.fps_label.config(
            text=f"FPS: {f:.1f}", foreground="green" if f > 15 else "orange"))
        self.root.after(0, lambda d=detections_data: self._populate_table(d))
        self.root.after(0, lambda c=len(detections_data): self._update_det_count_label(c))
        if detections_data and self.logger:
            self.root.after(0, lambda d=detections_data: self.logger.log_batch(d, 'video'))

    def _pause_video(self):
        self.inference.pause_video()
        self.play_btn.config(text=T('detect.play'))
        self.video_progress.config(state="normal")

    def _on_video_seek(self, value):
        target = int(float(value) * self.inference.total_frames / 100)
        result = self.inference.seek_video(target)
        if result:
            img, detections_data = result
            self._display_result_image(img)
            if detections_data:
                self._populate_table(detections_data)
                self._update_det_count_label(len(detections_data))

    def _start_video_generation(self):
        if not self.model_loader.is_loaded or not self.inference.video_path:
            return
        output = filedialog.asksaveasfilename(
            title=T('detect.gen_video'), defaultextension=".mp4",
            filetypes=[("MP4", "*.mp4"), ("AVI", "*.avi"), ("All", "*.*")]
        )
        if not output:
            return
        self.progress.start()
        self.video_gen_btn.config(state="disabled")

        def _run():
            try:
                ok, msg = self.inference.generate_annotated_video(output, on_progress=self._on_video_gen_progress)
                if ok:
                    self.root.after(0, lambda: messagebox.showinfo(T('dlg.done'), f"{T('dlg.saved_to')}\n{output}"))
                    self.root.after(0, lambda: self.status_var.set(T('detect.status_video_saved')))
                else:
                    self.root.after(0, lambda: messagebox.showerror(T('dlg.error'), msg))
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror(T('dlg.error'), str(e)))
            finally:
                self.root.after(0, self.progress.stop)
                self.root.after(0, lambda: self.video_gen_btn.config(state="normal"))

        threading.Thread(target=_run, daemon=True).start()

    def _on_video_gen_progress(self, current, total):
        p = int((current / total) * 100) if total > 0 else 0
        self.root.after(0, lambda: self.status_var.set(
            f"{T('detect.status_gen_video')} {p}% ({current}/{total})"))

    # ==================== CAMERA ====================
    def _refresh_cameras(self):
        cameras = self.inference.refresh_cameras()
        if cameras:
            self.camera_idx_combo['values'] = cameras
            if self.camera_idx_var.get() not in cameras:
                self.camera_idx_var.set(cameras[0])
            self.status_var.set(f"{T('detect.status_found_cam')} {len(cameras)}{T('detect.status_cam_unit')}")
        else:
            messagebox.showwarning(T('dlg.warning'), T('dlg.no_cameras'))

    def _toggle_camera(self):
        if not self.inference.camera_playing:
            try:
                idx = int(self.camera_idx_var.get())
            except ValueError:
                messagebox.showerror(T('dlg.error'), T('dlg.invalid_cam_idx'))
                return
            self._apply_confidence()
            ok, msg = self.inference.start_camera(idx, on_frame_callback=self._on_camera_frame_ext)
            if not ok:
                messagebox.showerror(T('dlg.error'), msg)
        else:
            self.inference.stop_camera()

    def _on_camera_frame_ext(self, pil_img, current_fps, detections_data, update_fps=False):
        if pil_img is None or pil_img.size[0] < 2:
            return
        self.root.after(0, lambda: self._display_result_image(pil_img))
        if update_fps:
            self.root.after(0, lambda f=current_fps: self.fps_label.config(
                text=f"FPS: {f:.1f}", foreground="green" if f > 15 else "orange"))
        self.root.after(0, lambda d=detections_data: self._populate_table(d))
        self.root.after(0, lambda c=len(detections_data): self._update_det_count_label(c))
        if detections_data and self.logger:
            self.root.after(0, lambda d=detections_data: self.logger.log_batch(d, 'camera'))

    def _on_camera_started(self, **kwargs):
        self.root.after(0, lambda: self.camera_btn.config(text=T('detect.cam_stop')))
        self._cam_connected_idx = kwargs.get('idx', '?')
        self.root.after(0, self._update_camera_status_label)

    def _on_camera_stopped(self, **kwargs):
        self.root.after(0, lambda: self.camera_btn.config(text=T('detect.cam_start')))
        self._cam_connected_idx = None
        self.root.after(0, self._update_camera_status_label)
        self.fps_label.config(text="FPS: --", foreground="gray")

    def _update_camera_status_label(self):
        if getattr(self, '_cam_connected_idx', None) is not None:
            idx = self._cam_connected_idx
            self.camera_status_label.config(
                text=f"{T('detect.cam_connected')} (idx={idx})", foreground="green")
        else:
            self.camera_status_label.config(
                text=T('detect.cam_disconnected'), foreground="gray")

    # ==================== LABEL STUDIO ====================
    def _toggle_ls_service(self):
        ls = self.app.get_module('label_studio')
        if not ls.flask_available:
            messagebox.showerror(T('dlg.error'), T('dlg.flask_missing'))
            return
        if not ls.is_running:
            ok, msg = ls.start_service()
            if not ok:
                messagebox.showerror(T('dlg.error'), msg)
        else:
            ls.stop_service()

    def _on_ls_started(self, **kwargs):
        self.root.after(0, lambda: self.ls_btn.config(text=T('detect.ls_stop')))
        self._ls_port = kwargs.get('port', 5000)
        self.root.after(0, self._update_ls_status_label)

    def _on_ls_stopped(self, **kwargs):
        self.root.after(0, lambda: self.ls_btn.config(text=T('detect.ls_start')))
        self._ls_port = None
        self.root.after(0, self._update_ls_status_label)

    def _update_ls_status_label(self):
        if getattr(self, '_ls_port', None) is not None:
            port = self._ls_port
            self.ls_status_label.config(
                text=f"{T('detect.ls_running')} (port {port})", foreground="green")
        else:
            self.ls_status_label.config(
                text=T('detect.ls_stopped'), foreground="gray")

    # ==================== EXPORT ====================
    def _get_current_detections(self):
        if self.image_paths and self.current_image_index < len(self.image_paths):
            path = self.image_paths[self.current_image_index]
            if path in self.detection_results:
                return self.inference.extract_detections(self.detection_results[path])
        return self.inference.last_detections or []

    def _export_json(self):
        dets = self._get_current_detections()
        if not dets:
            messagebox.showinfo(T('dlg.info'), T('dlg.no_records_export'))
            return
        path = filedialog.asksaveasfilename(
            title="Export JSON", defaultextension=".json",
            filetypes=[("JSON", "*.json"), ("All", "*.*")]
        )
        if path:
            img_path = self.image_paths[self.current_image_index] if self.image_paths else ''
            data = self.inference.export_json(dets, img_path)
            with open(path, 'w', encoding='utf-8') as f:
                f.write(data)
            messagebox.showinfo(T('dlg.done'), f"{T('dlg.exported_to')}\n{path}")

    def _export_csv(self):
        dets = self._get_current_detections()
        if not dets:
            messagebox.showinfo(T('dlg.info'), T('dlg.no_records_export'))
            return
        path = filedialog.asksaveasfilename(
            title="Export CSV", defaultextension=".csv",
            filetypes=[("CSV", "*.csv"), ("All", "*.*")]
        )
        if path:
            data = self.inference.export_csv(dets)
            with open(path, 'w', encoding='utf-8') as f:
                f.write(data)
            messagebox.showinfo(T('dlg.done'), f"{T('dlg.exported_to')}\n{path}")

    def _snapshot(self):
        img = getattr(self, '_current_pil_image', None)
        if img is None:
            self.status_var.set("No image to snapshot")
            return
        filepath = self.inference.snapshot(img)
        self.status_var.set(f"Snapshot saved: {filepath}")

    # ==================== SAVE ====================
    def _save_current(self):
        if not self.image_paths or self.current_image_index >= len(self.image_paths):
            return
        path = self.image_paths[self.current_image_index]
        if path not in self.detection_results:
            messagebox.showinfo(T('dlg.info'), T('dlg.not_detected_yet'))
            return
        output = filedialog.asksaveasfilename(
            title=T('detect.save_current'), defaultextension=".jpg",
            filetypes=[("JPEG", "*.jpg"), ("PNG", "*.png"), ("All", "*.*")],
            initialfile=f"detection_{os.path.basename(path)}"
        )
        if output:
            orig = self._orig_image_cache.get(path)
            if orig is None:
                orig = Image.open(path)
            result_img = self.inference.draw_detections(orig, self.detection_results[path])
            result_img.save(output)
            messagebox.showinfo(T('dlg.done'), f"{T('dlg.saved_to')}\n{output}")

    def _save_all(self):
        if not self.detection_results:
            return
        save_dir = filedialog.askdirectory(title=T('detect.save_all'))
        if save_dir:
            cnt = 0
            for path in self.image_paths:
                if path in self.detection_results:
                    orig = self._orig_image_cache.get(path)
                    if orig is None:
                        orig = Image.open(path)
                    result_img = self.inference.draw_detections(orig, self.detection_results[path])
                    base = os.path.splitext(os.path.basename(path))[0]
                    ext = os.path.splitext(path)[1]
                    result_img.save(os.path.join(save_dir, f"detection_{base}{ext}"))
                    cnt += 1
            messagebox.showinfo(T('dlg.done'),
                                f"{T('dlg.saved_count')} {cnt}{T('dlg.saved_unit')}\n{save_dir}")

    # ==================== HISTORY ====================
    def _on_history_updated(self, **kwargs):
        self.root.after(0, self._update_history_display)

    def _update_history_display(self):
        self.history_text.config(state='normal')
        self.history_text.delete('1.0', tk.END)
        if not self.history_mgr.records:
            self.history_text.insert(tk.END, f"{T('detect.no_records')}\n")
        else:
            for record in reversed(self.history_mgr.records[-15:]):
                ts = record['timestamp'].split(' ')[1]
                self.history_text.insert(tk.END, f"[{ts}] {record['image']}\n")
                objs = ', '.join([f"{k}:{v}" for k, v in record['objects'].items()])
                self.history_text.insert(tk.END, f"  +- {objs} (total {record['total']})\n\n")
        self.history_text.config(state='disabled')

    def _clear_history(self):
        if messagebox.askyesno(T('dlg.confirm'), T('dlg.clear_history')):
            self.history_mgr.clear()

    def _export_history(self):
        if not self.history_mgr.records:
            messagebox.showwarning(T('dlg.warning'), T('dlg.no_records_export'))
            return
        path = filedialog.asksaveasfilename(
            title=T('detect.hist_export'), defaultextension=".txt",
            filetypes=[("Text", "*.txt"), ("JSON", "*.json"), ("All", "*.*")]
        )
        if path:
            self.history_mgr.export(path)
            messagebox.showinfo(T('dlg.done'), f"{T('dlg.exported_to')}\n{path}")

    # ==================== MISC ====================
    def _clear_results(self):
        self.image_paths = []
        self.current_image_index = 0
        self.detection_results = {}
        self._orig_image_cache.clear()
        self.image_path_var.set("")
        self.video_path_var.set("")
        self.status_var.set(T('detect.status_cleared'))
        self._clear_table()
        self.image_label.configure(image="", text=T('detect.placeholder'))
        self.det_count_label.config(text="Det: 0")

        for btn in [self.prev_btn, self.next_btn, self.play_btn, self.pause_btn,
                     self.save_btn, self.save_all_btn, self.video_gen_btn]:
            btn.config(state="disabled")
        self.video_progress.config(state="disabled")
        self.index_label.config(text="0 / 0")

        self.inference.stop_camera()
        self.inference.stop_video()

    def _on_close(self):
        self.inference.stop_camera()
        self.inference.stop_video()
        ls = self.app.get_module('label_studio')
        if ls and ls.is_running:
            ls.stop_service()
        if self.logger:
            self.logger.force_flush()
        self._class_filter_vars.clear()
        self._class_filter_cbs.clear()
        self._orig_image_cache.clear()
        self._unbind_events()
        off_change(self._refresh_i18n)
        self.root.destroy()

    # ==================== LOGGER ====================
    def _on_log_interval_changed(self):
        if self.logger:
            self.logger.interval = self.log_interval_var.get()

    def _update_log_label(self):
        if not self.logger:
            return
        lines = self.logger.log_lines
        if lines > 0:
            self.log_info_label.config(
                text=f"Log: {lines} lines, {self.logger.log_size_mb}MB @ {self.logger.interval}s",
                foreground="green")
        else:
            self.log_info_label.config(text=T('detect.log_empty'), foreground="gray")

    def _flush_log(self):
        if self.logger:
            self.logger.force_flush()
            self._update_log_label()
            self.status_var.set(f"Log flushed: {self.logger.log_path}")
