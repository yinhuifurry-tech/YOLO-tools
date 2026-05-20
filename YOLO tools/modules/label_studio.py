import os
import json
import tempfile
import threading
import subprocess
import requests
import shutil
import glob
from .base import BaseModule
from framework.events import Events

try:
    from flask import Flask, request, jsonify
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False


class LabelStudioService(BaseModule):
    def __init__(self):
        super().__init__()
        self._running = False
        self._thread = None
        self.port = 5000

    @property
    def is_running(self):
        return self._running

    @property
    def model_loader(self):
        return self.get_module('model_loader')

    @property
    def flask_available(self):
        return FLASK_AVAILABLE

    def on_start(self):
        self.port = self.config.get('label_studio', 'port', 5000)

    def start_service(self):
        if not FLASK_AVAILABLE:
            return False, "Flask not installed"
        if not self.model_loader or not self.model_loader.is_loaded:
            return False, "No model loaded"

        self._running = True
        self._thread = threading.Thread(target=self._run_server, daemon=True)
        self._thread.start()
        self.emit(Events.LS_SERVICE_STARTED, port=self.port)
        return True, f"LS service started on port {self.port}"

    def stop_service(self):
        self._running = False
        try:
            import requests
            requests.post(f'http://127.0.0.1:{self.port}/shutdown', timeout=2)
        except Exception:
            pass
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=3)
        self.emit(Events.LS_SERVICE_STOPPED)

    def _run_server(self):
        app = Flask(__name__)
        service_ref = self

        @app.route('/shutdown', methods=['POST'])
        def shutdown():
            func = request.environ.get('werkzeug.server.shutdown')
            if func:
                func()
            return jsonify({"status": "shutting down"})

        @app.route('/predict', methods=['POST'])
        def predict():
            ml = service_ref.model_loader
            if not ml or not ml.is_loaded:
                return jsonify({"error": "Model not loaded"}), 500
            try:
                req_data = request.json
                image_url = req_data.get('image_url', '')
                resp = requests.get(image_url)
                if resp.status_code != 200:
                    return jsonify({"error": "Cannot download image"}), 400

                temp_path = os.path.join(tempfile.gettempdir(), 'temp_img.jpg')
                with open(temp_path, 'wb') as f:
                    f.write(resp.content)

                results = ml.predict(temp_path)
                predictions = []
                boxes = results[0].boxes if isinstance(results, list) else results.boxes
                names = results[0].names if isinstance(results, list) else results.names

                if boxes is not None:
                    for box in boxes:
                        conf = float(box.conf.item() if hasattr(box.conf, 'item') else box.conf)
                        if hasattr(box.xyxy, 'cpu'):
                            xyxy = box.xyxy[0].cpu().numpy() if hasattr(box.xyxy, '__getitem__') else box.xyxy.cpu().numpy()
                        else:
                            xyxy = box.xyxy
                        cls = int(box.cls.item() if hasattr(box.cls, 'item') else box.cls)
                        if conf > 0.5:
                            orig = results[0].orig_img if isinstance(results, list) else results.orig_img
                            prediction = {
                                "id": len(predictions),
                                "type": "rectanglelabels",
                                "value": {
                                    "x": float(xyxy[0]) / float(orig.shape[1]) * 100,
                                    "y": float(xyxy[1]) / float(orig.shape[0]) * 100,
                                    "width": float(xyxy[2] - xyxy[0]) / float(orig.shape[1]) * 100,
                                    "height": float(xyxy[3] - xyxy[1]) / float(orig.shape[0]) * 100,
                                    "rotation": 0,
                                    "rectanglelabels": [names.get(cls, f"class_{cls}")]
                                },
                                "to_name": "image",
                                "from_name": "label",
                                "image_rotation": 0,
                                "original_width": orig.shape[1],
                                "original_height": orig.shape[0],
                            }
                            predictions.append(prediction)

                response_data = {
                    "predictions": [{
                        "result": predictions,
                        "score": max([p.get('score', 0) for p in predictions], default=0),
                        "model_version": "unknown"
                    }]
                }
                os.remove(temp_path)
                return jsonify(response_data)
            except Exception as e:
                return jsonify({"error": str(e)}), 500

        app.run(host='0.0.0.0', port=self.port, debug=False, use_reloader=False, threaded=True)

    def deploy_label_studio(self, port=8080):
        try:
            r = subprocess.run(['docker', '--version'], capture_output=True, check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False, "Docker is not installed. Install Docker Desktop first."
        except Exception:
            return False, "Docker not found. Is Docker Desktop installed and in PATH?"

        try:
            r = subprocess.run(['docker', 'info'], capture_output=True, text=True)
            if r.returncode != 0:
                err_lower = r.stderr.lower()
                if 'dockerdesktoplinuxengine' in err_lower or 'pipe' in err_lower:
                    return False, (
                        "Docker Desktop is not fully started. Open Docker Desktop, "
                        "wait for the whale icon to stop animating, then try again."
                    )
                return False, "Docker daemon is not running. Start Docker Desktop first."
        except Exception:
            return False, "Cannot connect to Docker. Is Docker Desktop running?"

        try:
            r = subprocess.run(['docker', 'ps', '-a', '--filter', 'name=label-studio-instance', '--format', '{{.Names}}'],
                               capture_output=True, text=True)
            if 'label-studio-instance' in r.stdout:
                subprocess.run(['docker', 'rm', '-f', 'label-studio-instance'],
                               capture_output=True, timeout=30)

            cmd = ['docker', 'run', '-d', '-p', f'{port}:8080',
                   '-v', 'label-studio-data:/label-studio/data',
                   '--name', 'label-studio-instance',
                   'heartexlabs/label-studio:latest']
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            if result.returncode == 0:
                cid = result.stdout.strip()[:12]
                return True, f"Label Studio running at http://localhost:{port} (container: {cid})"
            else:
                err = result.stderr.strip()
                if 'Conflict' in err or 'already in use' in err:
                    return False, "Port or container name conflict. Try a different port or restart Docker."
                if 'pull' in err.lower() or 'not found' in err.lower():
                    return False, f"Cannot download Label Studio image. Check network.\n{err[:200]}"
                return False, err[:300]
        except subprocess.TimeoutExpired:
            return False, "Docker command timed out. Is Docker responsive?"
        except Exception as e:
            return False, str(e)

    def convert_to_yolo(self, json_path):
        if not json_path or not os.path.exists(json_path):
            return False, "File not found"

        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                annotations = json.load(f)

            output_dir = "converted_labels"
            os.makedirs(output_dir, exist_ok=True)

            for item in annotations:
                image_path = item.get('data', {}).get('image', '')
                if not image_path:
                    continue
                filename = os.path.basename(image_path)
                base = os.path.splitext(filename)[0]
                label_path = os.path.join(output_dir, f"{base}.txt")

                yolo_annotations = []
                for ann in item.get('annotations', []):
                    for result in ann.get('result', []):
                        if result.get('type') == 'rectanglelabels':
                            value = result.get('value', {})
                            x = value.get('x', 0)
                            y = value.get('y', 0)
                            w = value.get('width', 0)
                            h = value.get('height', 0)
                            x_center = (x + w / 2) / 100.0
                            y_center = (y + h / 2) / 100.0
                            w_norm = w / 100.0
                            h_norm = h / 100.0
                            yolo_annotations.append(f"0 {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}")

                with open(label_path, 'w', encoding='utf-8') as f:
                    for line in yolo_annotations:
                        f.write(line + '\n')

            return True, output_dir
        except Exception as e:
            return False, str(e)
