# YOLO AI Platform

基于运行框架 + 模块化架构的 YOLO 目标检测与训练一体化平台。(注意，目前运行python环境为python3.10)

## 架构

```
framework/          # 运行框架层
  app.py            # 应用生命周期、模块注册、依赖注入
  config.py         # 集中配置管理
  events.py         # 发布/订阅事件总线
  i18n.py           # 中英文国际化 (120+ 翻译条目)

modules/            # 功能模块层（可插拔）
  base.py           # BaseModule 基类
  model_loader.py   # 模型加载 (PyTorch + ONNX)
  inference.py      # 推理引擎 (批量/视频/摄像头/导出)
  training.py       # 训练引擎 (18 个超参数可配)
  converter.py      # PT → ONNX 转换
  dataset.py        # 数据集管理 (上传/组织/YAML)
  label_studio.py   # Label Studio 标注服务集成
  history.py        # 检测历史记录
  logger.py         # 结构化 JSONL 日志

gui/                # 界面层
  detection_window.py  # 目标检测界面 (TreeView表格/类别筛选/FPS/快捷键)
  training_window.py   # 训练与模型管理界面 (4标签页/预设/全部参数)

main.py             # 入口，模块组装启动
```

## 安装

```bash
pip install ultralytics opencv-python pillow numpy requests onnxruntime flask
# 或：启动后点"安装依赖"自动安装
```

## 启动

```bash
python main.py                  # 启动器（中英文切换）
python main.py --detection      # 直接进入检测界面
python main.py --training       # 直接进入训练界面
python main.py --lang en        # 英文界面
```

## 功能

### 目标检测

| 功能 | 说明 |
|------|------|
| 图像批量检测 | 选择多张图片，模型自动标注边界框和类别 |
| 视频检测 | 播放视频并实时标注，支持进度条拖动 |
| 摄像头实时检测 | 支持多个摄像头索引，实时 FPS 显示 |
| 标注视频生成 | 将视频逐帧检测并输出标注后的视频文件 |
| 类别筛选 | 按类别勾选，只显示关心的目标 |
| 置信度/IoU 滑块 | 实时调整检测阈值 |
| 检测结果表格 | TreeView 展示每条检测的类别/置信度/坐标，支持点击表头排序 |
| JSON/CSV 导出 | 单张或批量导出检测结果 |
| 截图 | 保存当前检测画面 |
| 键盘快捷键 | Space=播放, ← →=翻页, S=保存, E=导出, C=摄像头, Esc=清除 |
| 结构化日志 | JSONL 格式，每 N 秒记录类别汇总统计 |

### 模型训练

| 功能 | 说明 |
|------|------|
| 基础参数 | model, data, epochs, batch, imgsz, device, project, name |
| 优化器参数 | lr0, lrf, momentum, weight_decay |
| 数据增强 | HSV-H/S/V, flipud, fliplr, mosaic, mixup |
| 快速预设 | 默认/保守/激进/微调 4 种参数组合 |
| 数据集上传 | 支持 zip/tar/tar.gz 自动解压并组织为 YOLO 目录结构 |
| dataset.yaml 生成 | 自动扫描标签文件，提取类别信息生成配置 |
| 模型下载 | 从镜像站下载 yolo11n/s/m/l/x 系列预训练模型 |
| PT → ONNX 转换 | 支持 Ultralytics 导出和传统 torch.onnx 两种方式 |
| Label Studio 部署 | Docker 一键部署标注平台 |

### Label Studio 集成

- Flask ML 后端自动预标注
- Label Studio JSON → YOLO txt 格式转换
- Docker 部署，数据持久化

## 项目结构

```
├── main.py
├── framework/
│   ├── __init__.py
│   ├── app.py
│   ├── config.py
│   ├── events.py
│   └── i18n.py
├── modules/
│   ├── __init__.py
│   ├── base.py
│   ├── model_loader.py
│   ├── inference.py
│   ├── training.py
│   ├── converter.py
│   ├── dataset.py
│   ├── label_studio.py
│   ├── history.py
│   └── logger.py
├── gui/
│   ├── __init__.py
│   ├── detection_window.py
│   └── training_window.py
├── yolo_detection_gui.py       # [保留] 原始检测脚本
├── yolo_train_convert_no_web.py # [保留] 原始训练脚本
└── detection_log.jsonl          # 结构化检测日志
```

## 命令行参数

```
--detection    直接启动检测界面
--training     直接启动训练界面
--lang zh|en   界面语言 (默认 zh)
```

## 事件系统

模块间通过 EventBus 解耦通信：

```
MODEL_LOADED / MODEL_UNLOADED   模型加载/卸载
DETECTION_STARTED / COMPLETED   检测开始/完成
TRAINING_STARTED / COMPLETED    训练开始/完成
CAMERA_STARTED / STOPPED        摄像头开关
CONVERSION_STARTED / COMPLETED  模型转换开始/完成
LS_SERVICE_STARTED / STOPPED    Label Studio 服务开关
HISTORY_UPDATED                 检测历史更新
ERROR                           全局错误
```

## 日志格式

`detection_log.jsonl` (JSONL，每行一条记录)：

```json
{"ts": "2026-05-18 16:30:00", "type": "summary", "elapsed_s": 10.0, "classes": {"person": 15, "car": 8}, "total": 23}
```

日志间隔可在检测界面左下角 Spinbox 调节（1~300 秒）。
