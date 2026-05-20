import os
import glob
from .base import BaseModule
from framework.events import Events


class ModelConverter(BaseModule):
    def __init__(self):
        super().__init__()

    def scan_models(self, base_dir='runs'):
        pt_files = glob.glob(os.path.join(base_dir, '**', '*.pt'), recursive=True)
        models = []
        for pt_file in pt_files:
            if os.path.isfile(pt_file):
                models.append({
                    'path': pt_file,
                    'rel_path': os.path.relpath(pt_file),
                    'size_mb': round(os.path.getsize(pt_file) / (1024 * 1024), 2),
                    'time': os.path.getmtime(pt_file),
                    'directory': os.path.dirname(pt_file),
                })
        models.sort(key=lambda x: x['time'], reverse=True)
        return models

    def convert(self, model_path, output_path=None, opset_version=11):
        import torch
        try:
            import onnx
        except ImportError:
            return False, "ONNX not installed"

        if output_path is None:
            output_path = model_path.replace('.pt', '.onnx')

        self.emit(Events.CONVERSION_STARTED, path=model_path)
        self.log(f"Converting: {model_path}")

        try:
            from ultralytics import YOLO
            yolo_model = YOLO(model_path)
            yolo_model.export(
                format='onnx', dynamic=False, int8=False,
                half=False, opset=12, simplify=True
            )
            if os.path.exists(output_path):
                try:
                    onnx_model = onnx.load(output_path)
                    onnx.checker.check_model(onnx_model)
                except Exception:
                    pass
                self.emit(Events.CONVERSION_COMPLETED, output_path=output_path)
                return True, output_path
        except Exception as ultralytics_error:
            self.log(f"Ultralytics export failed: {ultralytics_error}, trying fallback...")

        try:
            model = self._load_model(model_path)
            model.eval()
            input_size = 640
            if hasattr(model, 'stride'):
                stride = int(model.stride.max()) if hasattr(model.stride, 'max') else 32
            else:
                stride = 32
            dummy_input = torch.randn(1, 3, input_size, input_size).to(torch.float32)

            for param in model.parameters():
                if param.dtype == torch.float16:
                    dummy_input = dummy_input.half()
                    break

            torch.onnx.export(
                model, dummy_input, output_path,
                export_params=True, verbose=False,
                opset_version=12, do_constant_folding=True,
                input_names=['images'], output_names=['output'],
                dynamic_axes={'images': {0: 'batch'}, 'output': {0: 'batch'}},
            )
            self.emit(Events.CONVERSION_COMPLETED, output_path=output_path)
            return True, output_path
        except Exception as e:
            self.emit(Events.ERROR, message=str(e))
            return False, str(e)

    def _load_model(self, model_path):
        import torch
        try:
            return torch.load(model_path, map_location='cpu', weights_only=False)
        except TypeError:
            return torch.load(model_path, map_location='cpu')

    def scan_and_convert_latest(self, base_dir='runs'):
        models = self.scan_models(base_dir)
        if not models:
            return False, "No models found"
        latest = max(models, key=lambda x: x['time'])
        return self.convert(latest['path'])

    def download_model(self, model_name='yolov8n.pt', destination='.'):
        import urllib.request
        mirror_urls = [
            f"https://github.com/ultralytics/assets/releases/download/v8.2.0/{model_name}",
            f"https://hf-mirror.com/ultralytics/{model_name}/resolve/main/{model_name}",
        ]
        for url in mirror_urls:
            try:
                filepath = os.path.join(destination, model_name)
                urllib.request.urlretrieve(url, filepath)
                return True, filepath
            except Exception:
                continue
        return False, "All mirrors failed"
