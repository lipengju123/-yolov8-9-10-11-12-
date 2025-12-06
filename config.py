#!/usr/bin/env python3
"""
配置文件 - 模型基准测试参数设置
"""

# 模型列表配置
# config.py
MODELS = {
    # ---------- YOLOv8 ----------
    "yolov8n": "YOLOv8-nano",
    "yolov8s": "YOLOv8-small",
    "yolov8m": "YOLOv8-medium",

    # ---------- RT-DETR ----------
    "rtdetr-l": "RT-DETR-L",
    "rtdetr-x": "RT-DETR-X",

    # ---------- YOLOv11 ----------
    "yolo11n": "YOLOv11-nano",
    "yolo11m": "YOLOv11-medium",
    "yolo11x": "YOLOv11-xlarge",

    # ---------- YOLOv12 ----------
    "yolo12n": "YOLOv12-nano",
    "yolo12s": "YOLOv12-small",
    "yolo12m": "YOLOv12-medium",

}
# 测试参数配置
TEST_CONFIG = {
    "data_path": r"D:\python xiangmu\ultralytics-main\pridect\coco.yaml",
    "img_size": 640,
    "batch_size": 8,
    "max_samples": 10,
    "num_speed_tests": 10
}

# 输出目录配置
OUTPUT_CONFIG = {
    "base_dir": "benchmark_results",
    "data_dir": "data",
    "plots_dir": "plots",
    "detailed_dir": "detailed_results"
}

# 可视化配置
VISUALIZATION_CONFIG = {
    "figsize": (12, 8),
    "dpi": 300,
    "font_family": "DejaVu Sans",
    "colors": ['steelblue', 'lightcoral', 'mediumseagreen', 'orange', 'purple']
}