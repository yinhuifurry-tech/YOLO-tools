import os
import sys
import subprocess
import glob
import importlib
from .base import BaseModule
from framework.events import Events


class TrainingEngine(BaseModule):
    def __init__(self):
        super().__init__()
        self._training = False
        self._epoch_progress = 0

    @property
    def is_training(self):
        return self._training

    def check_dependencies(self):
        return self._install_dependencies()

    def _install_dependencies(self):
        pip_cmd = self._get_pip_cmd()
        if not pip_cmd:
            return False

        packages = [
            ('ultralytics', 'ultralytics'),
            ('numpy', 'numpy'),
            ('opencv-python', 'cv2'),
            ('matplotlib', 'matplotlib'),
            ('pandas', 'pandas'),
            ('seaborn', 'seaborn'),
            ('psutil', 'psutil'),
            ('tqdm', 'tqdm'),
            ('requests', 'requests'),
            ('pillow', 'PIL'),
            ('pyyaml', 'yaml'),
            ('scipy', 'scipy'),
            ('tensorboard', 'tensorboard'),
            ('thop', 'thop'),
        ]
        results = []
        for pkg, imp_name in packages:
            try:
                importlib.import_module(imp_name)
            except ImportError:
                try:
                    subprocess.check_call(pip_cmd + ['install', pkg])
                    results.append((pkg, True))
                except subprocess.CalledProcessError:
                    results.append((pkg, False))
                continue
            results.append((pkg, True))

        try:
            import torch
        except ImportError:
            try:
                subprocess.check_call(pip_cmd + ['install', 'torch', 'torchvision', 'torchaudio',
                                                  '--index-url', 'https://download.pytorch.org/whl/cpu'])
            except subprocess.CalledProcessError:
                pass

        try:
            import onnxruntime
        except ImportError:
            try:
                subprocess.check_call(pip_cmd + ['install', 'onnxruntime'])
            except subprocess.CalledProcessError:
                pass

        failed = [pkg for pkg, ok in results if not ok]
        self.log(f"Dependencies: {len(results) - len(failed)}/{len(results)} installed")
        return len(failed) == 0

    def _get_pip_cmd(self):
        for cmd in [[sys.executable, '-m', 'pip'], ['pip'], ['pip3']]:
            try:
                subprocess.run(cmd + ['--version'], capture_output=True, check=True)
                return cmd
            except (subprocess.CalledProcessError, FileNotFoundError):
                continue
        return None

    def train(self, **kwargs):
        from ultralytics import YOLO

        cfg = self.config.section('training')
        cfg.update(kwargs)

        data_cfg = cfg.get('data', 'dataset.yaml')
        if not os.path.exists(data_cfg):
            dm = self.get_module('dataset')
            if dm:
                dataset_dir = data_cfg.replace('.yaml', '').replace('.yml', '')
                if not os.path.isdir(dataset_dir):
                    dataset_dir = 'dataset'
                ok, msg = dm.create_yaml(dataset_dir)
                if ok:
                    cfg['data'] = 'dataset.yaml'
                    self.log(f"Auto-created dataset.yaml from {dataset_dir}")
                else:
                    return False, f"Cannot create dataset config: {msg}"
            else:
                return False, f"Data config not found: {data_cfg}\nUpload a dataset and use 'Create dataset.yaml' first."

        self._training = True
        self.emit(Events.TRAINING_STARTED, config=cfg)

        try:
            model_path = cfg['model']
            if os.path.exists(model_path):
                fsize = os.path.getsize(model_path)
                if fsize < 1024 * 1024:
                    os.remove(model_path)
                    self.log(f"Removed corrupted model ({fsize} bytes): {model_path}")
                    return False, (
                        f"Model '{model_path}' is corrupted ({fsize} bytes). It was deleted.\n"
                        "Download again: Training tab → Model → Download → select yolov8n.pt"
                    )
                self.log(f"Model OK: {model_path} ({fsize / (1024*1024):.1f} MB)")
            else:
                self.log(f"Model not found locally: {model_path}")

            model = YOLO(cfg['model'])
            train_args = {
                'data': cfg['data'],
                'epochs': cfg.get('epochs', 100),
                'batch': cfg.get('batch', 8),
                'imgsz': cfg.get('imgsz', 640),
                'device': cfg.get('device', ''),
                'project': cfg.get('project', 'runs/train'),
                'name': cfg.get('name', 'exp'),
            }
            for key in ['lr0', 'lrf', 'momentum', 'weight_decay',
                        'hsv_h', 'hsv_s', 'hsv_v', 'flipud', 'fliplr',
                        'mosaic', 'mixup']:
                if key in cfg:
                    train_args[key] = cfg[key]

            model.train(**train_args)
            self.emit(Events.TRAINING_COMPLETED, success=True)
            return True, "Training completed"
        except Exception as e:
            err = str(e)
            if 'PytorchStreamReader' in err or 'zip archive' in err or 'central directory' in err:
                return False, (
                    f"Model file '{cfg['model']}' is corrupted/incomplete (interrupted download).\n"
                    "Fix: Training tab → Model → Download → select a model (e.g. yolov8n.pt)\n"
                    f"Detail: {err[:200]}"
                )
            self.emit(Events.ERROR, message=err)
            return False, err
        finally:
            self._training = False

    def _create_yaml(self, dataset_path):
        try:
            import yaml
        except ImportError:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'pyyaml'])
            import yaml

        train_img = os.path.join(dataset_path, 'train', 'images')
        val_img = os.path.join(dataset_path, 'val', 'images')
        if not os.path.exists(train_img) or not os.path.exists(val_img):
            return None

        labels_path = os.path.join(dataset_path, 'train', 'labels')
        nc = 1
        names = ['object']
        if os.path.exists(labels_path):
            label_files = glob.glob(os.path.join(labels_path, '*.txt'))
            class_ids = set()
            for lf in label_files:
                try:
                    with open(lf, 'r') as f:
                        for line in f:
                            parts = line.strip().split()
                            if parts:
                                class_ids.add(int(parts[0]))
                except Exception:
                    continue
            if class_ids:
                nc = len(class_ids)
                names = [f"class_{i}" for i in sorted(list(class_ids))]

        data_config = {
            'path': dataset_path,
            'train': 'train/images',
            'val': 'val/images',
            'nc': nc,
            'names': names,
        }
        yaml_path = 'dataset.yaml'
        with open(yaml_path, 'w', encoding='utf-8') as f:
            yaml.dump(data_config, f, default_flow_style=False, allow_unicode=True)
        return yaml_path
