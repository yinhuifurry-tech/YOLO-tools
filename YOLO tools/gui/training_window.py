import os
import sys
import io
import shutil
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from framework.events import Events
from framework.i18n import T, on_change, off_change

_SCALE = 1.0
try:
    if 'main' in sys.modules:
        _SCALE = sys.modules['main'].get_dpi_scale()
except Exception:
    pass


class TrainingWindow:
    def __init__(self, app):
        self.app = app
        self.root = tk.Toplevel()
        self.root.title(T('train.title'))
        w, h = int(1080 * _SCALE), int(700 * _SCALE)
        self.root.geometry(f"{w}x{h}")
        self.root.minsize(int(950 * _SCALE), int(620 * _SCALE))

        self.training_engine = app.get_module('training')
        self.converter = app.get_module('converter')
        self.dataset_mgr = app.get_module('dataset')
        self.label_studio = app.get_module('label_studio')

        self._tr_widgets = []
        self._bound_events = []

        self._init_training_vars()
        self._create_widgets()
        self._bind_events()

        on_change(self._refresh_i18n)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _t(self, key, default=''):
        return T(key, default)

    def _refresh_i18n(self):
        self.root.title(T('train.title'))
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

    def _frm(self, parent, key, **kw):
        frm = ttk.LabelFrame(parent, text=self._t(key), **kw)
        self._tr_widgets.append(lambda k=key: frm.configure(text=self._t(k)))
        return frm

    def _tab(self, notebook, key):
        frm = ttk.Frame(notebook, padding="10")
        notebook.add(frm, text=self._t(key))
        self._tr_widgets.append(lambda k=key, n=notebook, f=frm: n.tab(f, text=self._t(k)))
        return frm

    def _init_training_vars(self):
        cfg = self.app.config.section('training')
        defaults = {
            'model': 'yolov8n.pt', 'data': 'dataset.yaml',
            'epochs': '100', 'batch': '8', 'imgsz': '640',
            'device': '', 'project': 'runs/train', 'name': 'exp',
            'lr0': '0.0005', 'lrf': '0.01', 'momentum': '0.937',
            'weight_decay': '0.0005',
            'hsv_h': '0.015', 'hsv_s': '0.7', 'hsv_v': '0.4',
            'flipud': '0.0', 'fliplr': '0.5', 'mosaic': '1.0', 'mixup': '0.0',
        }
        for k, v in defaults.items():
            cfg_v = cfg.get(k, v)
            setattr(self, f'_tv_{k}', tk.StringVar(value=str(cfg_v)))

    def _bind_events(self):
        pairs = [
            (Events.TRAINING_STARTED, self._on_training_started),
            (Events.TRAINING_COMPLETED, self._on_training_completed),
            (Events.CONVERSION_STARTED, self._on_conversion_started),
            (Events.CONVERSION_COMPLETED, self._on_conversion_completed),
            (Events.ERROR, self._on_error),
        ]
        for ev, cb in pairs:
            self.app.events.on(ev, cb)
            self._bound_events.append((ev, cb))

    def _unbind_events(self):
        for ev, cb in self._bound_events:
            self.app.events.off(ev, cb)
        self._bound_events.clear()

    def _create_widgets(self):
        main_frame = ttk.Frame(self.root, padding="8")
        main_frame.pack(fill=tk.BOTH, expand=True)

        left_panel = ttk.Frame(main_frame)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        right_panel = ttk.Frame(main_frame, width=320)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=False)
        right_panel.pack_propagate(False)

        # Right Panel: Progress
        prog_frame = self._frm(right_panel, 'train.progress', padding="5")
        prog_frame.pack(fill=tk.X, pady=(0, 5))

        self._lbl(prog_frame, 'train.task_label').pack(anchor=tk.W)
        self.task_progress = ttk.Progressbar(prog_frame, length=200, mode='determinate')
        self.task_progress.pack(fill=tk.X, pady=(2, 5))
        self.task_label = ttk.Label(prog_frame, text=self._t('train.idle'), foreground="gray")
        self.task_label.pack(anchor=tk.W)

        # Right Panel: Log
        log_frame = self._frm(right_panel, 'train.log_title', padding="5")
        log_frame.pack(fill=tk.BOTH, expand=True)

        log_toolbar = ttk.Frame(log_frame)
        log_toolbar.pack(fill=tk.X, pady=(0, 3))
        self._btn(log_toolbar, 'train.log_clear', command=self._clear_log, width=6).pack(side=tk.RIGHT)
        self._btn(log_toolbar, 'train.log_export', command=self._export_log, width=10).pack(side=tk.RIGHT, padx=3)

        self.log_text = tk.Text(log_frame, wrap=tk.WORD, width=38, height=10, font=("Consolas", 9))
        log_scroll = ttk.Scrollbar(log_frame, orient=tk.VERTICAL, command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=log_scroll.set)
        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        log_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        self.status_var = tk.StringVar(value=self._t('train.ready'))
        ttk.Label(right_panel, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W).pack(
            fill=tk.X, pady=(5, 0))

        # Left Panel: Notebook
        notebook = ttk.Notebook(left_panel)
        notebook.pack(fill=tk.BOTH, expand=True)

        self._build_training_tab(notebook)
        self._build_data_tab(notebook)
        self._build_model_tab(notebook)
        self._build_deploy_tab(notebook)

    # ==================== TAB 1: Training ====================
    def _build_training_tab(self, notebook):
        tab = self._tab(notebook, 'train.tab_train')

        canvas = tk.Canvas(tab, highlightthickness=0)
        scrollbar = ttk.Scrollbar(tab, orient=tk.VERTICAL, command=canvas.yview)
        scroll_frame = ttk.Frame(canvas)
        scroll_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=scroll_frame, anchor=tk.NW)
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        canvas.bind("<MouseWheel>", _on_mousewheel, add='+')
        canvas.bind("<Enter>", lambda e: canvas.focus_set())

        row = 0

        # ---- Basic Parameters ----
        basic_frame = self._frm(scroll_frame, 'train.basic_params', padding="8")
        basic_frame.grid(row=row, column=0, sticky=(tk.W, tk.E), pady=(0, 8))
        basic_frame.columnconfigure(1, weight=1)

        fields_basic = [
            ('train.model_path', self._tv_model, 'train.browse_pt', self._browse_model),
            ('train.data_path', self._tv_data, 'detect.browse', self._browse_data),
            ('train.epochs', self._tv_epochs, None, None),
            ('train.batch', self._tv_batch, None, None),
            ('train.imgsz', self._tv_imgsz, None, None),
            ('train.device', self._tv_device, None, None),
            ('train.project', self._tv_project, 'train.browse_dir', lambda: self._browse_dir(self._tv_project)),
            ('train.name', self._tv_name, None, None),
        ]
        for i, (lbl_key, var, btn_key, btn_cmd) in enumerate(fields_basic):
            self._lbl(basic_frame, lbl_key, width=12, anchor=tk.E).grid(
                row=i, column=0, sticky=tk.W, padx=(0, 5), pady=2)
            ttk.Entry(basic_frame, textvariable=var).grid(
                row=i, column=1, sticky=(tk.W, tk.E), padx=(0, 5), pady=2)
            if btn_key and btn_cmd:
                self._btn(basic_frame, btn_key, command=btn_cmd, width=9).grid(row=i, column=2, pady=2)

        row += 1

        # ---- Optimizer ----
        opt_frame = self._frm(scroll_frame, 'train.optimizer', padding="8")
        opt_frame.grid(row=row, column=0, sticky=(tk.W, tk.E), pady=(0, 8))
        opt_frame.columnconfigure(1, weight=1)
        opt_frame.columnconfigure(3, weight=1)

        fields_opt = [
            ('train.lr0', self._tv_lr0), ('train.lrf', self._tv_lrf),
            ('train.momentum', self._tv_momentum), ('train.weight_decay', self._tv_weight_decay),
        ]
        for i, (lbl_key, var) in enumerate(fields_opt):
            col = (i % 2) * 2
            row_in = i // 2
            self._lbl(opt_frame, lbl_key, width=13, anchor=tk.E).grid(
                row=row_in, column=col, sticky=tk.W, padx=(0, 5), pady=2)
            ttk.Entry(opt_frame, textvariable=var, width=10).grid(
                row=row_in, column=col+1, sticky=tk.W, pady=2)

        row += 1

        # ---- Data Augmentation ----
        aug_frame = self._frm(scroll_frame, 'train.augmentation', padding="8")
        aug_frame.grid(row=row, column=0, sticky=(tk.W, tk.E), pady=(0, 8))
        aug_frame.columnconfigure(1, weight=1)
        aug_frame.columnconfigure(3, weight=1)

        fields_aug = [
            ('train.hsv_h', self._tv_hsv_h), ('train.hsv_s', self._tv_hsv_s),
            ('train.hsv_v', self._tv_hsv_v), ('train.flipud', self._tv_flipud),
            ('train.fliplr', self._tv_fliplr), ('train.mosaic', self._tv_mosaic),
            ('train.mixup', self._tv_mixup),
        ]
        for i, (lbl_key, var) in enumerate(fields_aug):
            col = (i % 2) * 2
            row_in = i // 2
            self._lbl(aug_frame, lbl_key, width=13, anchor=tk.E).grid(
                row=row_in, column=col, sticky=tk.W, padx=(0, 5), pady=2)
            ttk.Entry(aug_frame, textvariable=var, width=10).grid(
                row=row_in, column=col+1, sticky=tk.W, pady=2)

        row += 1

        # ---- Presets ----
        preset_frame = self._frm(scroll_frame, 'train.presets', padding="5")
        preset_frame.grid(row=row, column=0, sticky=(tk.W, tk.E), pady=(0, 8))

        self._btn(preset_frame, 'train.preset_default', command=self._preset_default).grid(row=0, column=0, padx=3, pady=2)
        self._btn(preset_frame, 'train.preset_conservative', command=self._preset_conservative).grid(row=0, column=1, padx=3, pady=2)
        self._btn(preset_frame, 'train.preset_aggressive', command=self._preset_aggressive).grid(row=0, column=2, padx=3, pady=2)
        self._btn(preset_frame, 'train.preset_finetune', command=self._preset_finetune).grid(row=0, column=3, padx=3, pady=2)

        row += 1

        # ---- Action Buttons ----
        action_frame = ttk.Frame(scroll_frame)
        action_frame.grid(row=row, column=0, sticky=(tk.W, tk.E), pady=(5, 0))

        self.start_train_btn = self._btn(action_frame, 'train.start', command=self._start_training)
        self.start_train_btn.pack(side=tk.LEFT, padx=(0, 10))
        self._btn(action_frame, 'train.install_deps', command=self._install_deps).pack(side=tk.LEFT, padx=(0, 10))
        self._btn(action_frame, 'train.reset_defaults', command=self._reset_training_vars).pack(side=tk.LEFT)
        self._btn(action_frame, 'train.save_default', command=self._save_as_default).pack(side=tk.RIGHT)

    def _preset_default(self):
        self._apply_preset({'epochs': '100', 'lr0': '0.0005', 'lrf': '0.01', 'momentum': '0.937',
                            'weight_decay': '0.0005', 'batch': '8', 'imgsz': '640',
                            'hsv_h': '0.015', 'hsv_s': '0.7', 'hsv_v': '0.4',
                            'flipud': '0.0', 'fliplr': '0.5', 'mosaic': '1.0', 'mixup': '0.0'})

    def _preset_conservative(self):
        self._apply_preset({'epochs': '200', 'lr0': '0.0001', 'lrf': '0.001', 'momentum': '0.95',
                            'weight_decay': '0.001', 'batch': '4', 'imgsz': '640',
                            'hsv_h': '0.01', 'hsv_s': '0.5', 'hsv_v': '0.3',
                            'flipud': '0.0', 'fliplr': '0.5', 'mosaic': '0.5', 'mixup': '0.1'})

    def _preset_aggressive(self):
        self._apply_preset({'epochs': '50', 'lr0': '0.001', 'lrf': '0.1', 'momentum': '0.9',
                            'weight_decay': '0.0001', 'batch': '16', 'imgsz': '320',
                            'hsv_h': '0.02', 'hsv_s': '0.9', 'hsv_v': '0.5',
                            'flipud': '0.1', 'fliplr': '0.5', 'mosaic': '1.0', 'mixup': '0.2'})

    def _preset_finetune(self):
        self._apply_preset({'epochs': '30', 'lr0': '0.0001', 'lrf': '0.0005', 'momentum': '0.937',
                            'weight_decay': '0.0005', 'batch': '4', 'imgsz': '640',
                            'hsv_h': '0.005', 'hsv_s': '0.3', 'hsv_v': '0.2',
                            'flipud': '0.0', 'fliplr': '0.3', 'mosaic': '0.0', 'mixup': '0.0'})

    def _apply_preset(self, vals):
        for k, v in vals.items():
            var = getattr(self, f'_tv_{k}', None)
            if var:
                var.set(v)

    def _reset_training_vars(self):
        self._init_training_vars()

    def _save_as_default(self):
        for k in ['model', 'data', 'epochs', 'batch', 'imgsz', 'device', 'project', 'name',
                  'lr0', 'lrf', 'momentum', 'weight_decay',
                  'hsv_h', 'hsv_s', 'hsv_v', 'flipud', 'fliplr', 'mosaic', 'mixup']:
            var = getattr(self, f'_tv_{k}', None)
            if var:
                self.app.config.set('training', k, var.get())

    def _start_training(self):
        try:
            kwargs = {
                'model': self._tv_model.get(), 'data': self._tv_data.get(),
                'epochs': int(self._tv_epochs.get()), 'batch': int(self._tv_batch.get()),
                'imgsz': int(self._tv_imgsz.get()), 'device': self._tv_device.get(),
                'project': self._tv_project.get(), 'name': self._tv_name.get(),
                'lr0': float(self._tv_lr0.get()), 'lrf': float(self._tv_lrf.get()),
                'momentum': float(self._tv_momentum.get()), 'weight_decay': float(self._tv_weight_decay.get()),
                'hsv_h': float(self._tv_hsv_h.get()), 'hsv_s': float(self._tv_hsv_s.get()),
                'hsv_v': float(self._tv_hsv_v.get()), 'flipud': float(self._tv_flipud.get()),
                'fliplr': float(self._tv_fliplr.get()), 'mosaic': float(self._tv_mosaic.get()),
                'mixup': float(self._tv_mixup.get()),
            }
        except ValueError as e:
            messagebox.showerror(T('dlg.error'), f"{T('dlg.invalid_value')}:\n{e}")
            return

        self._log("[*] Training parameters:")
        for k, v in kwargs.items():
            self._log(f"    {k}: {v}")

        self._run_in_thread(
            lambda: self.training_engine.train(**kwargs),
            "Training model", self.task_progress
        )

    def _browse_model(self):
        path = filedialog.askopenfilename(
            title=T('train.model_path'), filetypes=[("PyTorch", "*.pt"), ("All", "*.*")])
        if path:
            self._tv_model.set(path)

    def _browse_data(self):
        path = filedialog.askopenfilename(
            title=T('train.data_path'), filetypes=[("YAML", "*.yaml *.yml"), ("All", "*.*")])
        if path:
            self._tv_data.set(path)

    def _browse_dir(self, var=None):
        if var is None:
            var = self._tv_project
        path = filedialog.askdirectory(title=T('train.browse_dir'))
        if path:
            var.set(path)

    # ==================== TAB 2: Data Management ====================
    def _build_data_tab(self, notebook):
        tab = self._tab(notebook, 'train.tab_data')

        upload_frame = self._frm(tab, 'train.data_upload_group', padding="8")
        upload_frame.pack(fill=tk.X, pady=(0, 8))

        self._lbl(upload_frame, 'train.data_prompt_upload').pack(anchor=tk.W)
        btn_row = ttk.Frame(upload_frame)
        btn_row.pack(fill=tk.X, pady=(5, 0))
        self._btn(btn_row, 'train.upload_archive', command=self._upload_dataset).pack(side=tk.LEFT, padx=(0, 10))

        ttk.Separator(upload_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)

        self._lbl(upload_frame, 'train.data_prompt_org').pack(anchor=tk.W)
        org_row = ttk.Frame(upload_frame)
        org_row.pack(fill=tk.X, pady=(5, 0))
        self._lbl(org_row, 'train.split_ratio').pack(side=tk.LEFT, padx=(0, 5))
        self._tv_split_ratio = tk.StringVar(value="0.8")
        ttk.Spinbox(org_row, textvariable=self._tv_split_ratio, from_=0.5, to=1.0, increment=0.05, width=5).pack(
            side=tk.LEFT, padx=(0, 10))
        self._btn(org_row, 'train.organize_dataset', command=self._organize_dataset).pack(side=tk.LEFT)

        yaml_frame = self._frm(tab, 'train.data_yaml_group', padding="8")
        yaml_frame.pack(fill=tk.X, pady=(0, 8))

        self._lbl(yaml_frame, 'train.data_prompt_yaml').pack(anchor=tk.W)
        yaml_row = ttk.Frame(yaml_frame)
        yaml_row.pack(fill=tk.X, pady=(5, 0))
        self._btn(yaml_row, 'train.create_yaml', command=self._create_yaml).pack(side=tk.LEFT)

        ls_frame = self._frm(tab, 'train.data_ls_group', padding="8")
        ls_frame.pack(fill=tk.X, pady=(0, 8))

        self._lbl(ls_frame, 'train.data_prompt_ls').pack(anchor=tk.W)
        ls_row = ttk.Frame(ls_frame)
        ls_row.pack(fill=tk.X, pady=(5, 0))
        self._btn(ls_row, 'train.select_json_convert', command=self._convert_ls_data).pack(side=tk.LEFT)

    def _upload_dataset(self):
        path = filedialog.askopenfilename(
            title=T('train.upload_archive'),
            filetypes=[("Archives", "*.zip *.tar *.tar.gz"), ("All", "*.*")]
        )
        if not path:
            return
        self._run_in_thread(
            lambda: self.dataset_mgr.upload_archive(path),
            "Uploading dataset", self.task_progress
        )

    def _organize_dataset(self):
        try:
            ratio = float(self._tv_split_ratio.get())
        except ValueError:
            messagebox.showerror(T('dlg.error'), T('dlg.invalid_split'))
            return
        self._run_in_thread(
            lambda: self.dataset_mgr.organize(split_ratio=ratio),
            "Organizing dataset", self.task_progress
        )

    def _create_yaml(self):
        dir_path = filedialog.askdirectory(title=T('train.create_yaml'))
        if not dir_path:
            dir_path = "dataset"
        self._run_in_thread(
            lambda: self.dataset_mgr.create_yaml(dir_path),
            "Creating dataset.yaml", self.task_progress
        )

    def _convert_ls_data(self):
        path = filedialog.askopenfilename(
            title=T('train.select_json_convert'), filetypes=[("JSON", "*.json"), ("All", "*.*")])
        if not path:
            return
        self._run_in_thread(
            lambda: self.label_studio.convert_to_yolo(path),
            "Converting LS data", self.task_progress
        )

    # ==================== TAB 3: Model Management ====================
    def _build_model_tab(self, notebook):
        tab = self._tab(notebook, 'train.tab_model')

        dl_frame = self._frm(tab, 'train.model_dl_group', padding="8")
        dl_frame.pack(fill=tk.X, pady=(0, 8))
        dl_row = ttk.Frame(dl_frame)
        dl_row.pack(fill=tk.X)
        ttk.Label(dl_row, text="YOLO:").pack(side=tk.LEFT, padx=(0, 5))
        self._tv_dl_model = tk.StringVar(value="yolov8n.pt")
        ttk.Combobox(dl_row, textvariable=self._tv_dl_model, width=20,
                     values=["yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt",
                             "yolov8n-seg.pt", "yolov8s-seg.pt", "yolo11n.pt", "yolo11s.pt"]).pack(
            side=tk.LEFT, padx=(0, 10))
        self._btn(dl_row, 'train.download_model', command=self._download_model).pack(side=tk.LEFT)

        up_frame = self._frm(tab, 'train.model_up_group', padding="8")
        up_frame.pack(fill=tk.X, pady=(0, 8))
        self._btn(up_frame, 'train.upload_local', command=self._upload_model).pack(side=tk.LEFT)

        scan_frame = self._frm(tab, 'train.model_scan_group', padding="8")
        scan_frame.pack(fill=tk.X, pady=(0, 8))
        scan_row = ttk.Frame(scan_frame)
        scan_row.pack(fill=tk.X)
        self._lbl(scan_row, 'train.scan_dir').pack(side=tk.LEFT, padx=(0, 5))
        self._tv_scan_dir = tk.StringVar(value="runs")
        ttk.Entry(scan_row, textvariable=self._tv_scan_dir, width=15).pack(side=tk.LEFT, padx=(0, 10))
        self._btn(scan_row, 'train.scan_models', command=self._scan_models).pack(side=tk.LEFT)

        conv_frame = self._frm(tab, 'train.convert_group', padding="8")
        conv_frame.pack(fill=tk.X, pady=(0, 8))
        conv_row1 = ttk.Frame(conv_frame)
        conv_row1.pack(fill=tk.X, pady=(0, 5))
        self._btn(conv_row1, 'train.convert_latest', command=self._convert_latest).pack(side=tk.LEFT)

        conv_row2 = ttk.Frame(conv_frame)
        conv_row2.pack(fill=tk.X)
        self._btn(conv_row2, 'train.convert_selected', command=self._convert_selected).pack(side=tk.LEFT)

    def _download_model(self):
        name = self._tv_dl_model.get()
        self._run_in_thread(
            lambda: self.converter.download_model(name),
            f"Downloading {name}", self.task_progress
        )

    def _upload_model(self):
        path = filedialog.askopenfilename(
            title=T('train.upload_local'), filetypes=[("Models", "*.pt *.onnx"), ("All", "*.*")])
        if not path:
            return
        os.makedirs("models", exist_ok=True)
        dest = os.path.join("models", os.path.basename(path))
        shutil.copy2(path, dest)
        self._log(f"[+] Model uploaded to: {dest}")

    def _scan_models(self):
        directory = self._tv_scan_dir.get()
        self._run_in_thread(
            lambda: self.converter.scan_models(directory),
            f"Scanning {directory}", self.task_progress
        )

    def _convert_latest(self):
        self._run_in_thread(
            lambda: self.converter.scan_and_convert_latest(),
            "Converting latest model", self.task_progress
        )

    def _convert_selected(self):
        path = filedialog.askopenfilename(
            title=T('train.convert_selected'), filetypes=[("PyTorch", "*.pt"), ("All", "*.*")])
        if not path:
            return
        output = filedialog.asksaveasfilename(
            title=T('train.convert_selected'), defaultextension=".onnx",
            filetypes=[("ONNX", "*.onnx"), ("All", "*.*")])
        if not output:
            return
        self._run_in_thread(
            lambda: self.converter.convert(path, output),
            f"Converting {os.path.basename(path)}", self.task_progress
        )

    # ==================== TAB 4: Deploy ====================
    def _build_deploy_tab(self, notebook):
        tab = self._tab(notebook, 'train.tab_deploy')

        ls_frame = self._frm(tab, 'train.deploy_ls', padding="8")
        ls_frame.pack(fill=tk.X, pady=(0, 8))

        ttk.Label(ls_frame, text="Deploy Label Studio via Docker:").pack(anchor=tk.W, pady=(0, 5))
        ls_row = ttk.Frame(ls_frame)
        ls_row.pack(fill=tk.X)
        self._lbl(ls_row, 'train.ls_port').pack(side=tk.LEFT, padx=(0, 5))
        self._tv_ls_port = tk.StringVar(value="8080")
        ttk.Entry(ls_row, textvariable=self._tv_ls_port, width=8).pack(side=tk.LEFT, padx=(0, 10))
        self._btn(ls_row, 'train.deploy_ls', command=self._deploy_ls).pack(side=tk.LEFT)

        info_frame = self._frm(tab, 'train.ls_info_title', padding="8")
        info_frame.pack(fill=tk.BOTH, expand=True)

        info_label = ttk.Label(info_frame, text=self._t('train.ls_info_text'),
                                font=("Consolas", 9), justify=tk.LEFT)
        info_label.pack(anchor=tk.W, padx=5, pady=5)
        self._tr_widgets.append(lambda: info_label.config(text=T('train.ls_info_text')))

    def _deploy_ls(self):
        try:
            port = int(self._tv_ls_port.get())
        except ValueError:
            port = 8080
        self._run_in_thread(
            lambda: self.label_studio.deploy_label_studio(port),
            f"Deploying LS on port {port}", self.task_progress
        )

    # ==================== Utility ====================
    def _install_deps(self):
        self._run_in_thread(
            self.training_engine.check_dependencies,
            "Installing dependencies", self.task_progress
        )

    def _log(self, text):
        def _update():
            self.log_text.insert(tk.END, text + "\n")
            self.log_text.see(tk.END)
            if int(self.log_text.index('end-1c').split('.')[0]) > 2000:
                self.log_text.delete('1.0', '10.0')
        self.root.after(0, _update)

    def _clear_log(self):
        self.log_text.delete('1.0', tk.END)

    def _export_log(self):
        path = filedialog.asksaveasfilename(
            title=T('train.log_export'), defaultextension=".txt",
            filetypes=[("Text", "*.txt"), ("All", "*.*")])
        if path:
            with open(path, 'w', encoding='utf-8') as f:
                f.write(self.log_text.get('1.0', tk.END))
            self._log(f"[+] Log exported to: {path}")

    def _run_in_thread(self, target, status_text, progress_bar=None):
        def _wrapped():
            self.root.after(0, lambda: self.status_var.set(status_text))
            self.root.after(0, lambda: self.task_label.config(text=status_text, foreground="blue"))
            if progress_bar:
                self.root.after(0, lambda: progress_bar.config(mode='indeterminate'))
                self.root.after(0, progress_bar.start)
            self._log(f"{'='*50}")
            self._log(f"[*] {status_text}...")
            old_out, old_err = sys.stdout, sys.stderr

            class _DualOut:
                def __init__(s, orig, cb):
                    s.orig = orig; s.cb = cb
                def write(s, data):
                    s.orig.write(data)
                    if data.strip():
                        s.cb(data.rstrip())
                def flush(s):
                    s.orig.flush()

            captured = io.StringIO()
            dual = _DualOut(captured, self._log)
            try:
                sys.stdout = dual
                sys.stderr = dual
                result = target()
                if isinstance(result, tuple) and len(result) >= 2:
                    ok, msg = result
                    self._log(f"[+] {'Success' if ok else 'Failed'}: {msg}")
                elif isinstance(result, list):
                    self._log(f"[+] Found {len(result)} item(s)")
            except Exception as e:
                self._log(f"[!] Error: {e}")
            finally:
                sys.stdout = old_out
                sys.stderr = old_err
                if progress_bar:
                    self.root.after(0, progress_bar.stop)
                    self.root.after(0, lambda: progress_bar.config(mode='determinate', value=0))
                self.root.after(0, lambda: self.status_var.set(T('train.ready')))
                self.root.after(0, lambda: self.task_label.config(text=T('train.idle'), foreground="gray"))
                self._log(f"[*] {status_text} complete")
                self._log(f"{'='*50}")

        threading.Thread(target=_wrapped, daemon=True).start()

    # ===== Event Handlers =====
    def _on_training_started(self, **kwargs):
        self._log("[*] Training started")

    def _on_training_completed(self, **kwargs):
        self._log("[+] Training completed successfully!")

    def _on_conversion_started(self, **kwargs):
        self._log(f"[*] Conversion started: {kwargs.get('path', '')}")

    def _on_conversion_completed(self, **kwargs):
        self._log(f"[+] Conversion completed: {kwargs.get('output_path', '')}")

    def _on_error(self, **kwargs):
        self._log(f"[!] Error: {kwargs.get('message', 'Unknown')}")

    def _on_close(self):
        self._unbind_events()
        off_change(self._refresh_i18n)
        self.root.destroy()
