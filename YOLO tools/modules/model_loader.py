import os
import numpy as np
import cv2
from PIL import Image
from .base import BaseModule
from .detection_result import DetectionResult, extract_from_ultralytics, extract_from_onnx
from framework.events import Events


class ModelLoader(BaseModule):
    def __init__(self):
        super().__init__()
        self.model_path = None
        self.pytorch_model = None
        self.onnx_session = None
        self.use_onnx = False
        self.model_names = {}
        self.onnx_input_size = 640
        self._providers_available = []
        self._model_format = None

    @property
    def is_loaded(self):
        return self.pytorch_model is not None or self.onnx_session is not None

    @property
    def names(self):
        return self.model_names

    @property
    def num_classes(self):
        return len(self.model_names)

    @property
    def input_size(self):
        if self.use_onnx:
            return self.onnx_input_size
        return 640

    @property
    def model_type(self):
        return 'ONNX' if self.use_onnx else 'PyTorch'

    def load(self, file_path):
        self.model_path = file_path
        if file_path.endswith('.onnx'):
            self._model_format = 'ONNX'
            self._load_onnx(file_path)
        else:
            self._model_format = 'PyTorch'
            self._load_pytorch(file_path)
        self.emit(Events.MODEL_LOADED, path=file_path,
                  model_type=self._model_format, num_classes=self.num_classes)
        self.log(f"Model loaded: {os.path.basename(file_path)} [{self._model_format}]")

    def unload(self):
        self.model_path = None
        self.pytorch_model = None
        self.onnx_session = None
        self.use_onnx = False
        self.model_names = {}
        self._model_format = None
        self.emit(Events.MODEL_UNLOADED)

    def _load_pytorch(self, model_path):
        import torch
        from ultralytics import YOLO
        torch.backends.cudnn.enabled = False
        self.pytorch_model = YOLO(model_path, task='detect')
        self.use_onnx = False
        self.model_names = self.pytorch_model.names if hasattr(self.pytorch_model, 'names') and self.pytorch_model.names else {}
        if not self.model_names:
            raise RuntimeError(
                "Model loaded but has no class names. "
                "The file may be a training checkpoint (best.pt/last.pt is fine), "
                "a non-YOLO model, or corrupted. "
                "Try a fresh pretrained model like yolov8n.pt"
            )
        if not hasattr(self.pytorch_model, 'predict'):
            raise RuntimeError(
                "Loaded object is not a valid YOLO model (missing predict method). "
                "The file may contain model weights only (state_dict) rather than a full YOLO model. "
                "Use a .pt file exported by YOLO training (e.g. best.pt) or a pretrained model."
            )
        torch.set_num_threads(4)
        if hasattr(torch, 'set_float32_matmul_precision'):
            torch.set_float32_matmul_precision('high')
        self.log(f"PyTorch model: {len(self.model_names)} classes, input {self.input_size}x{self.input_size}")

    def _load_onnx(self, model_path):
        import onnxruntime as ort
        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        session_options.intra_op_num_threads = 4
        session_options.inter_op_num_threads = 4
        session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

        available_providers = ort.get_available_providers()
        self._providers_available = available_providers

        if 'CUDAExecutionProvider' in available_providers:
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        elif 'DmlExecutionProvider' in available_providers:
            providers = ['DmlExecutionProvider', 'CPUExecutionProvider']
        elif 'OpenVINOExecutionProvider' in available_providers:
            providers = ['OpenVINOExecutionProvider', 'CPUExecutionProvider']
        else:
            providers = ['CPUExecutionProvider']

        self.onnx_session = ort.InferenceSession(
            model_path, sess_options=session_options, providers=providers
        )
        self.use_onnx = True
        input_node = self.onnx_session.get_inputs()[0]
        self.onnx_input_size = input_node.shape[2]
        self.model_names = {i: f'class_{i}' for i in range(1000)}

    def predict(self, image):
        if not self.is_loaded:
            raise RuntimeError("No model loaded")
        try:
            if self.use_onnx and self.onnx_session:
                return self._predict_onnx(image)
            else:
                raw = self._predict_pytorch(image)
                return [extract_from_ultralytics(raw)]
        except Exception as e:
            img_desc = image if isinstance(image, str) else f"<{type(image).__name__}>"
            import traceback
            self.log(f"Predict FAILED | input: {img_desc} | error: {e}")
            self.log(f"Traceback:\n{traceback.format_exc()}")
            raise

    def _predict_pytorch(self, image):
        conf = self.config.get('model', 'conf_threshold', 0.35)
        iou = self.config.get('model', 'iou_threshold', 0.5)
        if isinstance(image, str):
            img = image.strip()
            if not img:
                raise ValueError("Empty image path string")
            if not os.path.exists(img):
                raise FileNotFoundError(f"Image not found: {img}")
            if not os.path.isfile(img):
                raise IsADirectoryError(f"Path is a directory, not a file: {img}")
            if os.path.getsize(img) == 0:
                raise ValueError(f"Image file is empty (0 bytes): {img}")
            if os.path.getsize(img) < 128:
                raise ValueError(f"File too small to be an image ({os.path.getsize(img)} bytes): {img}")
            try:
                from PIL import Image as PILImage
                test = PILImage.open(img)
                test.verify()
                w, h = test.size
                if w < 2 or h < 2:
                    raise ValueError(f"Image dimensions too small: {w}x{h}")
            except Exception as pe:
                raise ValueError(f"File is not a valid image: {img} ({pe})") from pe
            return self.pytorch_model.predict(img, save=False, verbose=False, conf=conf, iou=iou)
        elif isinstance(image, Image.Image):
            img_array = np.array(image)
            if len(img_array.shape) not in (2, 3):
                raise ValueError(f"Unexpected image shape: {img_array.shape}")
            if len(img_array.shape) == 3:
                if img_array.shape[2] == 4:
                    img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2BGR)
                elif img_array.shape[2] == 3:
                    img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            return self.pytorch_model.predict(img_array, save=False, verbose=False, conf=conf, iou=iou)
        elif isinstance(image, np.ndarray):
            if len(image.shape) not in (2, 3):
                raise ValueError(f"Unexpected ndarray shape: {image.shape}")
            if len(image.shape) == 3 and image.shape[2] == 4:
                image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
            return self.pytorch_model.predict(image, save=False, verbose=False, conf=conf, iou=iou)
        else:
            raise TypeError(f"Unsupported image type: {type(image)}")

    def _preprocess_onnx(self, image):
        h, w = image.shape[:2]
        target = self.onnx_input_size
        scale = min(target / w, target / h)
        new_w, new_h = int(w * scale), int(h * scale)
        img_resized = cv2.resize(image, (new_w, new_h))
        pad_w = target - new_w
        pad_h = target - new_h
        pad_left = pad_w // 2
        pad_top = pad_h // 2
        img_padded = cv2.copyMakeBorder(img_resized, pad_top, pad_h - pad_top,
                                         pad_left, pad_w - pad_left,
                                         cv2.BORDER_CONSTANT, value=(114, 114, 114))
        img_rgb = cv2.cvtColor(img_padded, cv2.COLOR_BGR2RGB)
        img_transposed = np.transpose(img_rgb, (2, 0, 1))
        img_normalized = img_transposed.astype(np.float32) / 255.0
        return np.expand_dims(img_normalized, axis=0), (scale, pad_left, pad_top)

    def _postprocess_onnx(self, output, original_shape, letterbox_params=None):
        conf_threshold = self.config.get('model', 'conf_threshold', 0.35)
        iou_threshold = self.config.get('model', 'iou_threshold', 0.5)
        predictions = np.squeeze(output[0])
        if predictions.ndim == 2 and predictions.shape[1] > predictions.shape[0]:
            predictions = predictions.T
        num_features = predictions.shape[0]
        boxes_raw = predictions[:4, :]
        if num_features == 85:
            obj_conf = predictions[4, :]
            class_scores = predictions[5:, :] * obj_conf
        else:
            class_scores = predictions[4:, :]
        max_scores = np.max(class_scores, axis=0)
        class_ids = np.argmax(class_scores, axis=0)
        keep = max_scores > conf_threshold
        boxes_raw = boxes_raw[:, keep]
        max_scores = max_scores[keep]
        class_ids = class_ids[keep]
        if boxes_raw.shape[1] == 0:
            return []
        cx, cy, w, h = boxes_raw[0], boxes_raw[1], boxes_raw[2], boxes_raw[3]
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2
        boxes = np.stack([x1, y1, x2, y2], axis=1)
        if letterbox_params:
            scale, pad_left, pad_top = letterbox_params
            boxes[:, [0, 2]] -= pad_left
            boxes[:, [1, 3]] -= pad_top
            boxes /= scale
        else:
            scale_x = original_shape[1] / self.onnx_input_size
            scale_y = original_shape[0] / self.onnx_input_size
            boxes[:, [0, 2]] *= scale_x
            boxes[:, [1, 3]] *= scale_y
        indices = cv2.dnn.NMSBoxes(boxes.tolist(), max_scores.tolist(), conf_threshold, iou_threshold)
        results = []
        if len(indices) > 0:
            for i in indices.flatten():
                results.append({
                    'box': boxes[i], 'score': float(max_scores[i]), 'class_id': int(class_ids[i])
                })
        return results

    def _predict_onnx(self, image):
        if isinstance(image, str):
            img = image.strip()
            if not img:
                raise ValueError("Empty image path string")
            if not os.path.exists(img):
                raise FileNotFoundError(f"Image not found: {img}")
            if os.path.getsize(img) == 0:
                raise ValueError(f"Image file is empty (0 bytes): {img}")
            if os.path.getsize(img) < 128:
                raise ValueError(f"File too small to be an image ({os.path.getsize(img)} bytes): {img}")
            pil_image = Image.open(img)
            pil_image.verify()
            pil_image = Image.open(img)  # re-open after verify
            img_array = np.array(pil_image)
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            original_shape = pil_image.size
        elif isinstance(image, Image.Image):
            img_array = np.array(image)
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            original_shape = image.size
        elif isinstance(image, np.ndarray):
            img_array = image
            original_shape = image.shape[:2]
        else:
            raise TypeError(f"Unsupported image type: {type(image)}")

        input_tensor, lb_params = self._preprocess_onnx(img_array)
        input_name = self.onnx_session.get_inputs()[0].name
        outputs = self.onnx_session.run(None, {input_name: input_tensor})
        detections = self._postprocess_onnx(outputs[0], original_shape, lb_params)
        return [extract_from_onnx(detections, self.model_names, img_array)]
