class DetectionBox:
    __slots__ = ('class_id', 'class_name', 'confidence', 'x1', 'y1', 'x2', 'y2', 'width', 'height')

    def __init__(self, class_id=0, class_name='', confidence=0.0,
                 x1=0.0, y1=0.0, x2=0.0, y2=0.0):
        self.class_id = int(class_id)
        self.class_name = str(class_name)
        self.confidence = round(float(confidence), 4)
        self.x1 = round(float(x1), 1)
        self.y1 = round(float(y1), 1)
        self.x2 = round(float(x2), 1)
        self.y2 = round(float(y2), 1)
        self.width = round(float(x2 - x1), 1)
        self.height = round(float(y2 - y1), 1)

    @property
    def xyxy(self):
        return [self.x1, self.y1, self.x2, self.y2]

    @property
    def conf(self):
        return self.confidence

    @property
    def cls(self):
        return self.class_id

    def __repr__(self):
        return (f"Box({self.class_name} conf={self.confidence:.2f} "
                f"@ [{self.x1:.0f},{self.y1:.0f} ~ {self.x2:.0f},{self.y2:.0f}])")

    def to_dict(self):
        return {
            'class_id': self.class_id, 'class_name': self.class_name,
            'confidence': self.confidence,
            'x1': self.x1, 'y1': self.y1, 'x2': self.x2, 'y2': self.y2,
            'width': self.width, 'height': self.height,
        }


class DetectionResult:
    __slots__ = ('boxes', 'names', 'image', 'image_path', 'fps')

    def __init__(self, boxes=None, names=None, image=None, image_path='', fps=0):
        self.boxes = boxes or []
        self.names = names or {}
        self.image = image
        self.image_path = image_path
        self.fps = fps

    def __len__(self):
        return len(self.boxes)

    def __iter__(self):
        return iter(self.boxes)

    def __getitem__(self, idx):
        return self.boxes[idx]

    @property
    def orig_img(self):
        import numpy as np
        if self.image is not None:
            return np.array(self.image)
        return np.zeros((640, 640, 3), dtype=np.uint8)


def extract_from_ultralytics(results):
    """Convert ultralytics results to DetectionResult."""
    if not results:
        return DetectionResult()
    r = results[0] if isinstance(results, list) else results
    boxes_src = r.boxes
    names = r.names if r.names else {}
    boxes = []
    if boxes_src is not None:
        for box in boxes_src:
            conf_val = _safe_val(box.conf)
            cls_val = _safe_val_int(box.cls)
            xyxy = _safe_xyxy(box.xyxy)
            if len(xyxy) >= 4:
                boxes.append(DetectionBox(
                    class_id=cls_val, class_name=names.get(cls_val, f'cls_{cls_val}'),
                    confidence=conf_val, x1=xyxy[0], y1=xyxy[1], x2=xyxy[2], y2=xyxy[3],
                ))
    return DetectionResult(boxes=boxes, names=names, image=r.orig_img if hasattr(r, 'orig_img') else None)


def extract_from_onnx(detections_list, names, image_array):
    """Convert ONNX postprocessing output to DetectionResult."""
    boxes = []
    for d in detections_list:
        b = d['box'] if isinstance(d, dict) else d
        boxes.append(DetectionBox(
            class_id=d.get('class_id', 0), class_name=names.get(d.get('class_id', 0), f"cls_{d.get('class_id', 0)}"),
            confidence=d.get('score', 0), x1=b[0], y1=b[1], x2=b[2], y2=b[3],
        ))
    return DetectionResult(boxes=boxes, names=names, image=image_array)


def _safe_val(val):
    try:
        v = val.item() if hasattr(val, 'item') else val
        if hasattr(v, '__len__') and not isinstance(v, (str, bytes)):
            v = v[0] if len(v) > 0 else 0
        return float(v)
    except (TypeError, ValueError, IndexError):
        return 0.0


def _safe_val_int(val):
    try:
        v = val.item() if hasattr(val, 'item') else val
        if hasattr(v, '__len__') and not isinstance(v, (str, bytes)):
            v = v[0] if len(v) > 0 else 0
        return int(v)
    except (TypeError, ValueError, IndexError):
        return 0


def _safe_xyxy(xyxy):
    if hasattr(xyxy, 'cpu'):
        a = xyxy.cpu().numpy()
        if a.ndim == 2 and a.shape[0] == 1:
            return a[0]
        return a
    if hasattr(xyxy, '__getitem__'):
        return xyxy
    import numpy as np
    return np.array(xyxy)
