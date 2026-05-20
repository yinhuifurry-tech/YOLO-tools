import os
import csv
import io
import time
import json
import threading
import queue
import cv2
import numpy as np
from PIL import Image
from collections import defaultdict
from .base import BaseModule
from framework.events import Events


class InferenceEngine(BaseModule):
    def __init__(self):
        super().__init__()
        self.target_fps = 60
        self.frame_interval = 1.0 / 60
        self.skip_frames = 0
        self.detection_interval = 2

        self._fps_buffer = []
        self._fps_max_samples = 30
        self._last_fps_time = time.time()
        self._frame_count_for_fps = 0
        self.current_fps = 0

        self._batch_results = {}
        self._is_processing = False
        self._stop_event = threading.Event()
        self._frame_queue = queue.Queue(maxsize=2)

        self.video_path = None
        self.video_cap = None
        self.total_frames = 0
        self.current_frame = 0
        self.playing = False

        self.camera_cap = None
        self.camera_playing = False
        self.camera_idx = 0

        self.class_filter = None
        self.class_colors = {}
        self._last_detections = None
        self._last_image_size = None
        self.box_line_width = 2
        self.font_scale = 0.6
        self.overlay_opacity = 1.0

    def on_start(self):
        cfg = self.config.section('inference')
        self.target_fps = cfg.get('target_fps', 60)
        self.frame_interval = 1.0 / self.target_fps
        self.skip_frames = cfg.get('skip_frames', 0)
        self.detection_interval = cfg.get('detection_interval', 2)
        self._frame_queue.maxsize = cfg.get('max_queue_size', 2)
        self.class_filter = set(cfg.get('class_filter', [])) if cfg.get('class_filter') else None

    def on_stop(self):
        self.stop_camera()
        self.stop_video()

    @property
    def model_loader(self):
        return self.get_module('model_loader')

    @property
    def fps(self):
        if len(self._fps_buffer) > 0:
            return sum(self._fps_buffer) / len(self._fps_buffer)
        return 0

    @property
    def last_detections(self):
        return self._last_detections

    def _tick_fps(self):
        now = time.time()
        self._frame_count_for_fps += 1
        elapsed = now - self._last_fps_time
        if elapsed >= 0.5:
            instant = self._frame_count_for_fps / elapsed
            self._fps_buffer.append(instant)
            if len(self._fps_buffer) > self._fps_max_samples:
                self._fps_buffer.pop(0)
            self.current_fps = self.fps
            self._frame_count_for_fps = 0
            self._last_fps_time = now

    def set_class_filter(self, class_ids):
        self.class_filter = set(class_ids) if class_ids else None

    def predict_image(self, image):
        if self.model_loader and self.model_loader.is_loaded:
            return self.model_loader.predict(image)
        raise RuntimeError("No model loaded")

    def _get_color(self, class_id):
        if class_id not in self.class_colors:
            import random
            rng = random.Random(class_id)
            self.class_colors[class_id] = tuple(rng.randint(64, 255) for _ in range(3))
        return self.class_colors[class_id]

    def _box_is_filtered(self, class_id):
        if self.class_filter is None:
            return False
        return class_id not in self.class_filter

    def draw_detections(self, image, results):
        img_array = np.array(image)
        if len(img_array.shape) == 3:
            img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        else:
            img_bgr = img_array

        boxes_src = results[0].boxes if isinstance(results, list) else results.boxes
        names = results[0].names if isinstance(results, list) else results.names
        conf_thresh = self.config.get('model', 'conf_threshold', 0.35)

        if boxes_src is not None:
            for box in boxes_src:
                conf_val = self._safe_float(box.conf)
                cls_val = self._safe_int(box.cls)
                if conf_val <= conf_thresh:
                    continue
                if self.class_filter is not None and cls_val not in self.class_filter:
                    continue

                xyxy = self._safe_array(box.xyxy)
                x1, y1, x2, y2 = map(int, xyxy[:4])
                color = self._get_color(cls_val)
                cv2.rectangle(img_bgr, (x1, y1), (x2, y2), color, self.box_line_width)
                label = f'{names.get(cls_val, f"cls_{cls_val}")} {conf_val:.2f}'
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, self.font_scale, 2)
                cv2.rectangle(img_bgr, (x1, max(0, y1 - th - 4)), (x1 + tw, y1), color, -1)
                cv2.putText(img_bgr, label, (x1, max(th, y1 - 2)),
                            cv2.FONT_HERSHEY_SIMPLEX, self.font_scale, (255, 255, 255), 2)

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        return Image.fromarray(img_rgb)

    def extract_detections(self, results):
        detections = []
        boxes_src = results[0].boxes if isinstance(results, list) else results.boxes
        names = results[0].names if isinstance(results, list) else results.names
        conf_thresh = self.config.get('model', 'conf_threshold', 0.35)

        if boxes_src is not None:
            for i, box in enumerate(boxes_src):
                conf_val = self._safe_float(box.conf)
                cls_val = self._safe_int(box.cls)
                if conf_val <= conf_thresh:
                    continue
                if self.class_filter is not None and cls_val not in self.class_filter:
                    continue
                xyxy = self._safe_array(box.xyxy)
                detections.append({
                    'id': i,
                    'class_id': cls_val,
                    'class_name': names.get(cls_val, f'class_{cls_val}'),
                    'confidence': round(conf_val, 4),
                    'x1': round(float(xyxy[0]), 1),
                    'y1': round(float(xyxy[1]), 1),
                    'x2': round(float(xyxy[2]), 1),
                    'y2': round(float(xyxy[3]), 1),
                    'width': round(float(xyxy[2] - xyxy[0]), 1),
                    'height': round(float(xyxy[3] - xyxy[1]), 1),
                })
        self._last_detections = detections
        return detections

    def class_summary(self, detections):
        summary = defaultdict(lambda: {'count': 0, 'avg_conf': 0.0})
        for d in detections:
            cn = d['class_name']
            summary[cn]['count'] += 1
            summary[cn]['avg_conf'] = (summary[cn]['avg_conf'] * (summary[cn]['count'] - 1) +
                                        d['confidence']) / summary[cn]['count']
        return dict(summary)

    def export_json(self, detections, image_path=None, pretty=True):
        data = {
            'image': image_path or '',
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'total': len(detections),
            'class_summary': {k: v['count'] for k, v in self.class_summary(detections).items()},
            'detections': detections,
        }
        return json.dumps(data, ensure_ascii=False, indent=2 if pretty else None)

    def export_csv(self, detections):
        if not detections:
            return 'id,class_id,class_name,confidence,x1,y1,x2,y2,width,height\n'
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(['id', 'class_id', 'class_name', 'confidence',
                          'x1', 'y1', 'x2', 'y2', 'width', 'height'])
        for d in detections:
            writer.writerow([d['id'], d['class_id'], d['class_name'], d['confidence'],
                              d['x1'], d['y1'], d['x2'], d['y2'], d['width'], d['height']])
        return output.getvalue()

    def snapshot(self, image):
        if not os.path.exists('snapshots'):
            os.makedirs('snapshots')
        filename = time.strftime('snapshot_%Y%m%d_%H%M%S.jpg')
        filepath = os.path.join('snapshots', filename)
        if isinstance(image, Image.Image):
            image.save(filepath)
        else:
            cv2.imwrite(filepath, cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR))
        return filepath

    # ---- Batch Detection ----
    def batch_detect(self, image_paths):
        self._is_processing = True
        self._batch_results = {}
        total = len(image_paths)
        self.emit(Events.DETECTION_STARTED, total=total)

        results = {}
        for idx, path in enumerate(image_paths):
            if not self._is_processing:
                break
            try:
                det = self.predict_image(path)
                results[path] = det
                self._batch_results[path] = det
                self.emit(Events.TRAINING_PROGRESS, current=idx + 1, total=total)
            except Exception as e:
                self.emit(Events.ERROR, message=str(e))

        self._is_processing = False
        self.emit(Events.DETECTION_COMPLETED, results=results)
        return results

    def get_batch_result(self, image_path):
        return self._batch_results.get(image_path)

    # ---- Video ----
    def load_video(self, path):
        self.video_path = path
        self.video_cap = cv2.VideoCapture(path)
        self.total_frames = int(self.video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.current_frame = 0
        return True

    def play_video(self, on_frame_callback=None):
        if not self.video_cap:
            return
        self.playing = True
        self._stop_event.clear()
        fps = self.video_cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30
        delay = 1.0 / fps

        def _loop():
            last_display = 0.0
            while self.playing and not self._stop_event.is_set():
                ret, frame = self.video_cap.read()
                if not ret:
                    cf_pos = int(self.video_cap.get(cv2.CAP_PROP_POS_FRAMES))
                    if cf_pos >= self.total_frames - 1:
                        self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    break

                frame = frame.copy()
                self._tick_fps()
                self.current_frame = int(self.video_cap.get(cv2.CAP_PROP_POS_FRAMES))

                # time-based throttle: max 25 display updates/sec (40ms interval)
                now = time.time()
                if now - last_display < 0.04:
                    time.sleep(delay)
                    continue
                last_display = now

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(frame_rgb)

                detections_data = []
                if self.model_loader and self.model_loader.is_loaded:
                    try:
                        results = self.predict_image(pil_img)
                        detections_data = self.extract_detections(results)
                        pil_img = self.draw_detections(pil_img, results)
                    except Exception:
                        pass

                if on_frame_callback:
                    on_frame_callback(pil_img, self.current_frame, self.total_frames,
                                       fps, self.current_fps, detections_data)
                time.sleep(delay)

        self._vid_thread = threading.Thread(target=_loop, daemon=True)
        self._vid_thread.start()

    def pause_video(self):
        self.playing = False
        self._stop_event.set()

    def stop_video(self):
        self.pause_video()
        if self.video_cap:
            self.video_cap.release()
            self.video_cap = None
        if hasattr(self, '_vid_thread') and self._vid_thread is not None:
            self._vid_thread.join(timeout=2)
            self._vid_thread = None

    def seek_video(self, target_frame):
        if self.video_cap:
            self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
            self.current_frame = target_frame
            ret, frame = self.video_cap.read()
            if ret:
                frame = frame.copy()
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(frame_rgb)
                detections_data = []
                if self.model_loader and self.model_loader.is_loaded:
                    try:
                        results = self.predict_image(pil_img)
                        detections_data = self.extract_detections(results)
                        pil_img = self.draw_detections(pil_img, results)
                    except Exception:
                        pass
                return pil_img, detections_data
        return None, []

    def generate_annotated_video(self, output_path, on_progress=None):
        if not self.video_path:
            return False, "No video loaded"
        cap = cv2.VideoCapture(self.video_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

        fn = 0
        while fn < total:
            ret, frame = cap.read()
            if not ret:
                break
            frame = frame.copy()
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(frame_rgb)
            try:
                results = self.predict_image(pil_img)
                pil_img = self.draw_detections(pil_img, results)
            except Exception:
                pass
            out.write(cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR))
            fn += 1
            if on_progress:
                on_progress(fn, total)

        cap.release()
        out.release()
        return True, output_path

    # ---- Camera ----
    def start_camera(self, idx=0, on_frame_callback=None):
        self.camera_idx = idx
        self.camera_cap = None
        for backend in [cv2.CAP_DSHOW, cv2.CAP_ANY, cv2.CAP_MSMF]:
            try:
                cap = cv2.VideoCapture(idx, backend)
                if cap.isOpened():
                    ret, frame = cap.read()
                    if ret and frame is not None and frame.size > 0:
                        self.camera_cap = cap
                        break
                    cap.release()
            except Exception:
                continue

        if self.camera_cap is None or not self.camera_cap.isOpened():
            return False, f"Cannot open camera {idx}. Try a different index."

        self.camera_cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.camera_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.camera_cap.set(cv2.CAP_PROP_FPS, 30)
        self.camera_cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.camera_playing = True
        self._stop_event.clear()

        def _loop():
            grab_errors = 0
            last_display = 0.0
            fps_update_counter = 0
            while self.camera_playing and not self._stop_event.is_set():
                if self.camera_cap is None or not self.camera_cap.isOpened():
                    break
                ret, frame = self.camera_cap.read()
                if not ret or frame is None or frame.size == 0:
                    grab_errors += 1
                    if grab_errors > 30:
                        break
                    time.sleep(0.05)
                    continue
                grab_errors = 0
                frame = frame.copy()
                h, w = frame.shape[:2]
                if w < 10 or h < 10:
                    time.sleep(0.5)
                    continue

                self._tick_fps()
                fps_update_counter += 1

                # time-based throttle: max 20 display updates/sec (50ms interval)
                now = time.time()
                if now - last_display < 0.05:
                    continue
                last_display = now

                try:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                except cv2.error:
                    continue

                pil_img = Image.fromarray(frame_rgb)
                detections_data = []
                try:
                    results = self.predict_image(pil_img)
                    detections_data = self.extract_detections(results)
                    pil_img = self.draw_detections(pil_img, results)
                except Exception:
                    pass

                current_fps_val = self.current_fps
                update_label = fps_update_counter >= 15
                if update_label:
                    fps_update_counter = 0
                if on_frame_callback:
                    on_frame_callback(pil_img, current_fps_val, detections_data, update_label)

        self._cam_thread = threading.Thread(target=_loop, daemon=True)
        self._cam_thread.start()
        self.emit(Events.CAMERA_STARTED, idx=idx)
        return True, f"Camera {idx} started"

    def stop_camera(self):
        self.camera_playing = False
        self._stop_event.set()
        if self.camera_cap:
            self.camera_cap.release()
            self.camera_cap = None
        if hasattr(self, '_cam_thread') and self._cam_thread is not None:
            self._cam_thread.join(timeout=2)
            self._cam_thread = None
        self.emit(Events.CAMERA_STOPPED)

    def refresh_cameras(self, max_idx=5):
        available = []
        for i in range(max_idx):
            for backend in [cv2.CAP_DSHOW, cv2.CAP_ANY]:
                try:
                    cap = cv2.VideoCapture(i, backend)
                    if cap.isOpened():
                        ret, _ = cap.read()
                        if ret and str(i) not in available:
                            available.append(str(i))
                        cap.release()
                        break
                except Exception:
                    pass
        return available

    def refresh_cameras_up_to(self, target_idx):
        scan = max(target_idx + 1, 5)
        return self.refresh_cameras(scan)

    # ---- Static helpers ----
    @staticmethod
    def _safe_float(val):
        try:
            v = val.item() if hasattr(val, 'item') else val
            if hasattr(v, '__len__') and not isinstance(v, (str, bytes)):
                v = v[0] if len(v) > 0 else 0
            return float(v)
        except (TypeError, ValueError, IndexError):
            return 0.0

    @staticmethod
    def _safe_int(val):
        try:
            v = val.item() if hasattr(val, 'item') else val
            if hasattr(v, '__len__') and not isinstance(v, (str, bytes)):
                v = v[0] if len(v) > 0 else 0
            return int(v)
        except (TypeError, ValueError, IndexError):
            return 0

    @staticmethod
    def _safe_array(xyxy):
        if hasattr(xyxy, 'cpu'):
            return xyxy[0].cpu().numpy() if hasattr(xyxy, '__getitem__') and len(
                xyxy.shape) > 1 else xyxy.cpu().numpy()
        elif hasattr(xyxy, '__getitem__') and hasattr(xyxy, 'shape'):
            if len(xyxy.shape) > 1:
                return xyxy[0]
            return xyxy
        return np.array(xyxy)
