_lang = 'zh'
_callbacks = []
_strings = {
    # ===== Launcher / Main =====
    'launcher.title':         {'zh': 'YOLO AI 平台 - 启动器',     'en': 'YOLO AI Platform - Launcher'},
    'launcher.subtitle':      {'zh': '运行框架 + 模块化架构',      'en': 'Runtime Framework + Modular Architecture'},
    'launcher.open_detect':   {'zh': '打开检测界面',               'en': 'Open Detection GUI'},
    'launcher.open_train':    {'zh': '打开训练界面',               'en': 'Open Training GUI'},
    'launcher.quick_check':   {'zh': '模块状态:',                  'en': 'Quick Check:'},
    'launcher.status_ok':     {'zh': '模型加载: 就绪 | 推理: 就绪 | 训练: 就绪', 'en': 'Model Loader: OK | Inference: OK | Training: OK'},
    'launcher.quit':          {'zh': '退出',                       'en': 'Quit'},
    'launcher.language':      {'zh': '语言',                       'en': 'Language'},

    # ===== Detection Window =====
    'detect.title':           {'zh': 'YOLO 目标检测',             'en': 'YOLO Object Detection'},
    'detect.model_group':     {'zh': '模型选择',                   'en': 'Model Selection'},
    'detect.model_label':     {'zh': '模型:',                      'en': 'Model:'},
    'detect.browse':          {'zh': '浏览',                       'en': 'Browse'},
    'detect.unload':          {'zh': '卸载',                       'en': 'Unload'},
    'detect.inference':       {'zh': '推理:',                      'en': 'Inference:'},
    'detect.conf':            {'zh': '置信度:',                    'en': 'Conf:'},
    'detect.file_group':      {'zh': '文件选择',                   'en': 'File Selection'},
    'detect.images':          {'zh': '图像:',                      'en': 'Images:'},
    'detect.video':           {'zh': '视频:',                      'en': 'Video:'},
    'detect.select_images':   {'zh': '选择图像',                   'en': 'Select Images'},
    'detect.select_video':    {'zh': '选择视频',                   'en': 'Select Video'},

    'detect.det_group':       {'zh': '检测',                       'en': 'Detection'},
    'detect.batch_detect':    {'zh': '批量检测',                   'en': 'Batch Detect'},
    'detect.gen_video':       {'zh': '生成标注视频',               'en': 'Generate Annotated Video'},

    'detect.cam_group':       {'zh': '摄像头',                     'en': 'Camera'},
    'detect.cam_index':       {'zh': '索引:',                      'en': 'Index:'},
    'detect.cam_refresh':     {'zh': '刷新',                       'en': 'Refresh'},
    'detect.cam_start':       {'zh': '开启摄像头',                 'en': 'Start Camera'},
    'detect.cam_stop':        {'zh': '关闭摄像头',                 'en': 'Stop Camera'},
    'detect.cam_disconnected':{'zh': '未连接',                     'en': 'disconnected'},
    'detect.cam_connected':   {'zh': '已连接',                     'en': 'connected'},

    'detect.ls_group':        {'zh': '标注服务',                   'en': 'Label Studio'},
    'detect.ls_start':        {'zh': '启动LS服务',                 'en': 'Start LS Service'},
    'detect.ls_stop':         {'zh': '停止LS服务',                 'en': 'Stop LS Service'},
    'detect.ls_stopped':      {'zh': '未启动',                     'en': 'stopped'},
    'detect.ls_running':      {'zh': '运行中',                     'en': 'running'},

    'detect.playback':        {'zh': '视频播放',                   'en': 'Playback'},
    'detect.play':            {'zh': '播放',                       'en': 'Play'},
    'detect.pause':           {'zh': '暂停',                       'en': 'Pause'},
    'detect.nav_group':       {'zh': '图像导航',                   'en': 'Image Navigation'},
    'detect.prev':            {'zh': '<< 上一张',                  'en': '<< Prev'},
    'detect.next':            {'zh': '下一张 >>',                  'en': 'Next >>'},

    'detect.output_group':    {'zh': '输出',                       'en': 'Output'},
    'detect.save_current':    {'zh': '保存当前',                   'en': 'Save Current'},
    'detect.save_all':        {'zh': '保存全部',                   'en': 'Save All'},
    'detect.clear_all':       {'zh': '全部清除',                   'en': 'Clear All'},

    'detect.result_group':    {'zh': '检测结果',                   'en': 'Detection Results'},
    'detect.hist_group':      {'zh': '检测历史',                   'en': 'Detection History'},
    'detect.hist_clear':      {'zh': '清空',                       'en': 'Clear'},
    'detect.hist_export':     {'zh': '导出',                       'en': 'Export'},

    # New: toolbar & table
    'detect.export_json':     {'zh': '导出JSON',                   'en': 'Export JSON'},
    'detect.export_csv':      {'zh': '导出CSV',                    'en': 'Export CSV'},
    'detect.snapshot':        {'zh': '截图',                       'en': 'Snapshot'},
    'detect.confidence':      {'zh': '置信度:',                    'en': 'Conf:'},
    'detect.iou':             {'zh': 'IoU:',                       'en': 'IoU:'},
    'detect.shortcuts_hint':  {'zh': '快捷键: Space=播放 ←→=导航 S=保存 E=导出 Esc=清除',
                               'en': 'Keys: Space=Play ←→=Nav S=Save E=Export Esc=Clear'},
    'detect.class_filter':    {'zh': '类别筛选',                   'en': 'Class Filter'},
    'detect.select_all':      {'zh': '全选',                       'en': 'All'},
    'detect.deselect_all':    {'zh': '全不选',                     'en': 'None'},
    'detect.tbl_class':       {'zh': '类别',                       'en': 'Class'},
    'detect.tbl_conf':        {'zh': '置信度',                     'en': 'Conf'},
    'detect.tbl_bbox':        {'zh': '边界框',                     'en': 'BBox'},
    'detect.class_count':     {'zh': '类别数:',                    'en': 'Classes:'},

    # Logger
    'detect.log_group':       {'zh': '日志记录',                   'en': 'Logging'},
    'detect.log_interval':    {'zh': '间隔:',                      'en': 'Interval:'},
    'detect.log_unit':        {'zh': '秒',                        'en': 's'},
    'detect.log_flush':       {'zh': '立即记录',                   'en': 'Flush Now'},
    'detect.log_empty':       {'zh': '日志: 暂无记录',              'en': 'Log: empty'},

    'detect.placeholder':     {'zh': '请选择模型和文件开始',        'en': 'Select model and files to start'},
    'detect.status_ready':    {'zh': '就绪 - 请选择模型开始',       'en': 'Ready - select a model to begin'},
    'detect.status_loaded':   {'zh': '模型已加载:',                'en': 'Model loaded:'},
    'detect.status_unloaded': {'zh': '模型已卸载',                 'en': 'Model unloaded'},
    'detect.status_detecting':{'zh': '正在检测中...',              'en': 'Detecting...'},
    'detect.status_done':     {'zh': '检测完成',                   'en': 'Detection complete'},
    'detect.status_cleared':  {'zh': '已清除 - 请重新选择文件',    'en': 'Cleared - select new files'},
    'detect.status_video_saved':{'zh': '标注视频已保存',           'en': 'Annotated video saved'},
    'detect.status_gen_video':{'zh': '正在生成视频...',            'en': 'Generating video...'},
    'detect.status_found_cam':{'zh': '发现',                       'en': 'Found'},
    'detect.status_cam_unit': {'zh': '个摄像头',                   'en': ' camera(s)'},
    'detect.detect_stats':    {'zh': '检测统计:',                  'en': 'Detection Statistics:'},
    'detect.total_objects':   {'zh': '共检测到:',                  'en': 'Total:'},
    'detect.objects_unit':    {'zh': '个对象',                     'en': ' objects'},
    'detect.no_objects':      {'zh': '未检测到任何对象',            'en': 'No objects detected'},
    'detect.not_detected':    {'zh': '尚未检测',                   'en': 'Not yet detected'},
    'detect.no_records':      {'zh': '暂无记录',                   'en': 'No records'},

    # ===== Training Window =====
    'train.title':            {'zh': 'YOLO 训练与模型管理',        'en': 'YOLO Training & Model Management'},
    'train.tab_train':        {'zh': '训练',                       'en': 'Training'},
    'train.tab_data':         {'zh': '数据',                       'en': 'Data'},
    'train.tab_model':        {'zh': '模型',                       'en': 'Model'},
    'train.tab_deploy':       {'zh': '部署',                       'en': 'Deploy'},

    'train.basic_params':     {'zh': '基础参数',                   'en': 'Basic Parameters'},
    'train.model_path':       {'zh': '模型文件:',                  'en': 'Model:'},
    'train.data_path':        {'zh': '数据配置:',                  'en': 'Data:'},
    'train.epochs':           {'zh': '训练轮数:',                  'en': 'Epochs:'},
    'train.batch':            {'zh': '批次大小:',                  'en': 'Batch:'},
    'train.imgsz':            {'zh': '图像尺寸:',                  'en': 'Img Size:'},
    'train.device':           {'zh': '设备:',                      'en': 'Device:'},
    'train.project':          {'zh': '项目目录:',                  'en': 'Project:'},
    'train.name':             {'zh': '运行名称:',                  'en': 'Name:'},
    'train.browse_pt':        {'zh': '选择PT',                     'en': 'Browse PT'},
    'train.browse_dir':       {'zh': '选择目录',                   'en': 'Browse Dir'},

    'train.optimizer':        {'zh': '优化器参数',                 'en': 'Optimizer Parameters'},
    'train.lr0':              {'zh': '初始学习率:',                'en': 'Initial LR:'},
    'train.lrf':              {'zh': '最终学习率因子:',            'en': 'Final LR Factor:'},
    'train.momentum':         {'zh': '动量:',                      'en': 'Momentum:'},
    'train.weight_decay':     {'zh': '权重衰减:',                  'en': 'Weight Decay:'},

    'train.augmentation':     {'zh': '数据增强',                   'en': 'Data Augmentation'},
    'train.hsv_h':            {'zh': '色相增强:',                  'en': 'HSV-H:'},
    'train.hsv_s':            {'zh': '饱和度增强:',                'en': 'HSV-S:'},
    'train.hsv_v':            {'zh': '明度增强:',                  'en': 'HSV-V:'},
    'train.flipud':           {'zh': '上下翻转:',                  'en': 'Flip UD:'},
    'train.fliplr':           {'zh': '左右翻转:',                  'en': 'Flip LR:'},
    'train.mosaic':           {'zh': '马赛克拼图:',                'en': 'Mosaic:'},
    'train.mixup':            {'zh': '混合增强:',                  'en': 'MixUp:'},

    'train.presets':          {'zh': '快速预设',                   'en': 'Quick Presets'},
    'train.preset_default':   {'zh': '默认(均衡)',                 'en': 'Default (Balanced)'},
    'train.preset_conservative':{'zh': '保守(低过拟合)',           'en': 'Conservative (Low Overfit)'},
    'train.preset_aggressive':{'zh': '激进(快速)',                 'en': 'Aggressive (Fast)'},
    'train.preset_finetune':  {'zh': '微调',                       'en': 'Fine-Tune'},

    'train.start':            {'zh': '开始训练',                   'en': 'START TRAINING'},
    'train.install_deps':     {'zh': '安装依赖',                   'en': 'Install Dependencies'},
    'train.reset_defaults':   {'zh': '恢复默认',                   'en': 'Reset to Defaults'},
    'train.save_default':     {'zh': '保存为默认',                 'en': 'Save as Default'},

    'train.upload_archive':   {'zh': '上传数据集压缩包',            'en': 'Upload Dataset Archive'},
    'train.organize_dataset': {'zh': '组织数据集',                  'en': 'Organize Dataset'},
    'train.split_ratio':      {'zh': '划分比例:',                  'en': 'Split ratio:'},
    'train.create_yaml':      {'zh': '创建YAML配置文件',            'en': 'Create Dataset Config YAML'},
    'train.import_ls':        {'zh': '导入Label Studio标注',        'en': 'Import Label Studio Annotations'},
    'train.select_json_convert':{'zh': '选择JSON并转换',            'en': 'Select JSON & Convert'},

    'train.download_model':   {'zh': '下载模型',                   'en': 'Download Model'},
    'train.upload_local':     {'zh': '上传本地模型',               'en': 'Upload Local Model'},
    'train.scan_models':      {'zh': '扫描模型',                   'en': 'Scan Models'},
    'train.scan_dir':         {'zh': '目录:',                      'en': 'Directory:'},
    'train.convert_latest':   {'zh': '自动转换最新模型',           'en': 'Auto-Convert Latest Model'},
    'train.convert_selected': {'zh': '选择PT文件转换',             'en': 'Select PT File to Convert'},
    'train.convert_group':    {'zh': '模型转换 (PT -> ONNX)',      'en': 'Model Conversion (PT -> ONNX)'},

    'train.deploy_ls':        {'zh': '部署Label Studio',           'en': 'Deploy Label Studio'},
    'train.ls_port':          {'zh': '端口:',                      'en': 'Port:'},
    'train.ls_info_title':    {'zh': '部署说明',                   'en': 'Deployment Info'},
    'train.ls_info_text':     {'zh':
        'Label Studio 部署说明:\n'
        '  - 需要安装并运行 Docker\n'
        '  - 默认访问地址: http://localhost:8080\n'
        '  - 首次访问: 创建管理员账户\n'
        '  - 数据存储在 Docker 卷: label-studio-data\n\n'
        '手动命令:\n'
        '  docker ps                           # 查看运行中的容器\n'
        '  docker stop label-studio-instance    # 停止服务\n'
        '  docker start label-studio-instance   # 重启服务',
        'en':
        'Label Studio Deployment:\n'
        '  - Requires Docker to be installed and running\n'
        '  - Default access: http://localhost:8080\n'
        '  - First access: create admin account\n'
        '  - Data persists in Docker volume: label-studio-data\n\n'
        'Commands (manual):\n'
        '  docker ps                           # Check running containers\n'
        '  docker stop label-studio-instance    # Stop service\n'
        '  docker start label-studio-instance   # Restart service'},

    'train.progress':         {'zh': '进度',                       'en': 'Progress'},
    'train.task_label':       {'zh': '任务:',                      'en': 'Task:'},
    'train.idle':             {'zh': '空闲',                       'en': 'Idle'},
    'train.log_clear':        {'zh': '清空',                       'en': 'Clear'},
    'train.log_export':       {'zh': '导出日志',                   'en': 'Export Log'},
    'train.log_title':        {'zh': '日志输出',                   'en': 'Log Output'},
    'train.ready':            {'zh': '就绪',                       'en': 'Ready'},

    'train.data_upload_group':{'zh': '上传与组织数据集',            'en': 'Upload & Organize Dataset'},
    'train.data_yaml_group':  {'zh': '创建数据集配置',              'en': 'Create Dataset Config'},
    'train.data_ls_group':    {'zh': 'Label Studio 数据导入',      'en': 'Label Studio Data Import'},
    'train.data_prompt_upload':{'zh': '上传压缩数据集文件 (zip, tar, tar.gz):', 'en': 'Upload compressed dataset archive (zip, tar, tar.gz):'},
    'train.data_prompt_org':  {'zh': '将现有的 images/ 和 labels/ 目录组织为数据集结构:', 'en': 'Organize existing images/ and labels/ into dataset structure:'},
    'train.data_prompt_yaml': {'zh': '从数据集目录生成 dataset.yaml:', 'en': 'Generate dataset.yaml from structured dataset directory:'},
    'train.data_prompt_ls':   {'zh': '将 Label Studio 导出的 JSON 转换为 YOLO 格式:', 'en': 'Convert Label Studio exported JSON to YOLO format:'},

    'train.model_dl_group':   {'zh': '下载模型',                   'en': 'Download Model'},
    'train.model_up_group':   {'zh': '上传本地模型',                'en': 'Upload Local Model'},
    'train.model_scan_group': {'zh': '扫描模型文件',                'en': 'Scan & List Models'},
    'train.model_conv_group': {'zh': '转换 PT 到 ONNX',            'en': 'Convert PT to ONNX'},

    # ===== Dialogs =====
    'dlg.error':              {'zh': '错误',                       'en': 'Error'},
    'dlg.warning':            {'zh': '警告',                       'en': 'Warning'},
    'dlg.info':               {'zh': '提示',                       'en': 'Info'},
    'dlg.confirm':            {'zh': '确认',                       'en': 'Confirm'},
    'dlg.done':               {'zh': '完成',                       'en': 'Done'},
    'dlg.no_model_image':     {'zh': '请先选择模型和图像文件',      'en': 'Select model and images first'},
    'dlg.no_model_video':     {'zh': '请先选择模型和视频文件',      'en': 'Select model and video first'},
    'dlg.no_model':           {'zh': '请先选择模型',               'en': 'Select a model first'},
    'dlg.load_failed':        {'zh': '模型加载失败',               'en': 'Failed to load model'},
    'dlg.img_load_failed':    {'zh': '图像加载失败',               'en': 'Failed to load image'},
    'dlg.no_cameras':         {'zh': '未找到可用的摄像头',          'en': 'No cameras found'},
    'dlg.invalid_cam_idx':    {'zh': '无效的摄像头索引',            'en': 'Invalid camera index'},
    'dlg.cam_start_failed':   {'zh': '摄像头启动失败',              'en': 'Failed to start camera'},
    'dlg.flask_missing':      {'zh': 'Flask 未安装, 请运行: pip install flask', 'en': 'Flask not installed, run: pip install flask'},
    'dlg.no_records_export':  {'zh': '暂无记录可导出',              'en': 'No records to export'},
    'dlg.clear_history':      {'zh': '确定要清空所有检测记录吗?',   'en': 'Clear all detection records?'},
    'dlg.saved_to':           {'zh': '已保存到:',                  'en': 'Saved to:'},
    'dlg.exported_to':        {'zh': '已导出到:',                  'en': 'Exported to:'},
    'dlg.saved_count':        {'zh': '已保存',                     'en': 'Saved'},
    'dlg.saved_unit':         {'zh': '个结果到:',                  'en': ' results to:'},
    'dlg.not_detected_yet':   {'zh': '尚未检测, 无法保存',          'en': 'Not yet detected'},
    'dlg.invalid_value':      {'zh': '参数值无效',                  'en': 'Invalid value'},
    'dlg.invalid_split':      {'zh': '无效的划分比例',              'en': 'Invalid split ratio'},
}


def tr(key, default=''):
    val = _strings.get(key, {})
    return val.get(_lang, val.get('en', default or key))


def T(key, default=''):
    return tr(key, default)


def set_lang(lang):
    global _lang
    _lang = lang
    for cb in _callbacks:
        try:
            cb()
        except Exception:
            pass


def on_change(callback):
    _callbacks.append(callback)


def off_change(callback):
    if callback in _callbacks:
        _callbacks.remove(callback)


def current():
    return _lang
