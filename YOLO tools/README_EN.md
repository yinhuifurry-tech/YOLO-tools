# YOLO AI Platform

A modular YOLO object detection and training platform with a runtime framework architecture, supporting both Chinese and English UI.

## Architecture

```
framework/          # Runtime framework layer
  app.py            # App lifecycle, module registry, dependency injection
  config.py         # Centralized configuration management
  events.py         # Publish/subscribe event bus
  i18n.py           # Chinese/English internationalization (120+ entries)

modules/            # Business logic modules (pluggable)
  model_loader.py   # Model loading (PyTorch + ONNX)
  inference.py      # Inference engine (batch / video / camera / export)
  training.py       # Training engine (18 configurable hyperparameters)
  converter.py      # PT → ONNX conversion
  dataset.py        # Dataset management (upload / organize / YAML generation)
  label_studio.py   # Label Studio annotation service integration
  history.py        # Detection history tracking
  logger.py         # Structured JSONL logging

gui/                # UI layer
  detection_window.py  # Detection GUI (TreeView table / class filter / FPS / shortcuts)
  training_window.py   # Training GUI (4 tabs / presets / full parameters)

main.py             # Entry point, module wiring, DPI-aware rendering
```

## Installation

```bash
pip install ultralytics opencv-python pillow numpy requests onnxruntime flask
# Or: launch the app and click "Install Dependencies" to auto-install
```

## Launch

```bash
python main.py                  # Launcher (language switcher)
python main.py --detection      # Open detection window directly
python main.py --training       # Open training window directly
python main.py --lang en        # English UI
```

## Features

### Object Detection

| Feature | Description |
|---------|-------------|
| Batch image detection | Multi-image batch processing with bounding boxes and labels |
| Video detection | Real-time playback with annotation overlay and seek support |
| Camera real-time | Multi-camera support with FPS counter |
| Annotated video export | Frame-by-frame detection output to video file |
| Class filter | Checkbox panel to show/hide specific classes |
| Confidence / IoU sliders | Real-time threshold adjustment |
| Results table | TreeView with sortable columns (class, confidence, bbox) |
| JSON / CSV export | Per-image or batch export |
| Snapshot | Save current detection frame |
| Keyboard shortcuts | Space=Play, Arrows=Nav, S=Save, E=Export, C=Camera, Esc=Clear |
| Structured logging | JSONL format, periodic class summary every N seconds |

### Model Training

| Feature | Description |
|---------|-------------|
| Basic parameters | model, data, epochs, batch, imgsz, device, project, name |
| Optimizer | lr0, lrf, momentum, weight_decay |
| Data augmentation | HSV-H/S/V, flipud, fliplr, mosaic, mixup |
| Quick presets | Default / Conservative / Aggressive / Fine-Tune |
| Dataset upload | Extract zip/tar/tar.gz and auto-organize into YOLO structure |
| YAML generation | Auto-scan labels to generate dataset.yaml |
| Model download | Download from mirrors (yolo11n/s/m/l/x series) |
| PT → ONNX conversion | Ultralytics export and traditional torch.onnx methods |
| Label Studio deploy | One-click Docker deployment |

### Label Studio Integration

- Flask ML backend for auto-annotation
- Label Studio JSON → YOLO txt format conversion
- Docker deployment with persistent storage

## Project Structure

```
├── main.py
├── framework/
│   ├── __init__.py  /  app.py  /  config.py  /  events.py  /  i18n.py
├── modules/
│   ├── __init__.py  /  base.py
│   ├── model_loader.py  /  inference.py  /  training.py
│   ├── converter.py  /  dataset.py  /  label_studio.py
│   ├── history.py  /  logger.py
├── gui/
│   ├── __init__.py  /  detection_window.py  /  training_window.py
├── yolo_detection_gui.py        # [legacy] original detection script
├── yolo_train_convert_no_web.py  # [legacy] original training script
├── README.md
└── detection_log.jsonl           # structured detection log
```

## CLI Arguments

```
--detection    Launch detection GUI directly
--training     Launch training GUI directly
--lang zh|en   UI language (default: zh)
```

## Event System

Modules communicate via a decoupled EventBus:

```
MODEL_LOADED / MODEL_UNLOADED      Model load/unload
DETECTION_STARTED / COMPLETED      Detection lifecycle
TRAINING_STARTED / COMPLETED       Training lifecycle
CAMERA_STARTED / STOPPED           Camera on/off
CONVERSION_STARTED / COMPLETED     Model conversion
LS_SERVICE_STARTED / STOPPED       Label Studio service
HISTORY_UPDATED                    Detection history
ERROR                              Global error
```

## Log Format

`detection_log.jsonl` (JSONL, one record per line):

```json
{"ts": "2026-05-18 16:30:00", "type": "summary", "elapsed_s": 10.0, "classes": {"person": 15, "car": 8}, "total": 23}
```

Log interval is adjustable via the Spinbox in the detection window (1–300 seconds).
