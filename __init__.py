#!/usr/bin/env python3
import os
import sys
import cv2
import torch
import threading
import time
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
from pathlib import Path
from datetime import datetime
from PIL import Image, ImageTk
from ultralytics import YOLO, __version__ as ultralytics_version

# ---------- 配置参数 ----------
CONFIDENCE_THRESHOLD = 0.3
LOG_UPDATE_INTERVAL = 0.33  # 每秒更新3次日志

MODEL_HUB = {
    "yolov8n": ("YOLOv8-nano (最快)", "yolov8n.pt"),
    "yolov8s": ("YOLOv8-small",           "yolov8s.pt"),
    "yolov8m": ("YOLOv8-medium",          "yolov8m.pt"),
    "rtdetr-l":("RT-DETR-L (平衡)",       "rtdetr-l.pt"),
    "rtdetr-x":("RT-DETR-X (最精确)",     "rtdetr-x.pt"),
    # ---------- YOLOv11 系列 ----------
    "yolo11n": ("YOLOv11-nano (11-最快)", "yolo11n.pt"),
    "yolo11m": ("YOLOv11-medium", "yolo11m.pt"),
    "yolo11x": ("YOLOv11-xlarge (11-最准)", "yolo11x.pt"),

    # ---------- YOLOv12 系列 ----------
    "yolo12n": ("YOLOv12-nano (12-最快)", "yolo12n.pt"),
    "yolo12m": ("YOLOv12-medium", "yolo12m.pt"),
}

# COCO 类别名称
COCO_NAMES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
    "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book",
    "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]


# ---------- 检测器类 ----------
class Detector:
    def __init__(self, model_name, device, confidence_threshold, output_dir):
        self.model = YOLO(model_name)
        self.device = device
        self.confidence_threshold = confidence_threshold
        self.output_dir = output_dir
        self.is_video_detection = False
        self.video_capture = None
        self.video_writer = None
        self.detection_thread = None
        self.stop_event = threading.Event()
        self.last_log_time = 0
        self.frame_count = 0
        self.video_interrupted = False

    def detect_image(self, image_path):
        """检测单张图像"""
        # 使用指定的设备进行推理
        results = self.model(image_path, device=self.device, verbose=False)[0]
        output_path = self.draw_detections(image_path, results)
        return results, output_path

    def detect_video(self, video_path, update_callback, log_callback):
        """检测视频流"""
        self.is_video_detection = True
        self.stop_event.clear()
        self.video_interrupted = False

        # 打开视频文件
        self.video_capture = cv2.VideoCapture(video_path)
        if not self.video_capture.isOpened():
            log_callback(f"[ERROR] 无法打开视频文件: {video_path}")
            return False

        fps = self.video_capture.get(cv2.CAP_PROP_FPS)
        frame_count = int(self.video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(self.video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

        log_callback(f"[INFO] 视频信息: {fps:.2f} FPS, 总帧数: {frame_count}, 分辨率: {width}x{height}")

        # 准备视频输出
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        video_name = Path(video_path).stem
        output_video_path = os.path.join(self.output_dir, f"{video_name}_detected_{timestamp}.mp4")

        # 定义视频编码器和写入器
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

        if not self.video_writer.isOpened():
            log_callback(f"[ERROR] 无法创建输出视频文件: {output_video_path}")
            return False

        log_callback(f"[INFO] 输出视频将保存到: {output_video_path}")

        # 视频检测线程
        self.detection_thread = threading.Thread(
            target=self._video_detection_loop,
            args=(update_callback, log_callback, output_video_path)
        )
        self.detection_thread.daemon = True
        self.detection_thread.start()

        return True

    def _video_detection_loop(self, update_callback, log_callback, output_video_path):
        """视频检测循环"""
        self.frame_count = 0
        self.last_log_time = time.time()

        while not self.stop_event.is_set() and self.video_capture.isOpened():
            ret, frame = self.video_capture.read()
            if not ret:
                break

            # 进行目标检测 - 使用指定的设备
            results = self.model(frame, device=self.device, verbose=False)[0]
            self.frame_count += 1

            # 绘制检测结果
            processed_frame = self.draw_detections_on_frame(frame, results)

            # 写入处理后的帧到输出视频
            self.video_writer.write(processed_frame)

            # 更新UI
            current_time = time.time()
            if current_time - self.last_log_time >= LOG_UPDATE_INTERVAL:
                detections = self._extract_detection_info(results)
                log_callback(f"[视频] 帧 {self.frame_count}: 检测到 {len(detections)} 个对象")
                for cls, count in detections.items():
                    log_callback(f"  - {COCO_NAMES[cls]}: {count}")
                self.last_log_time = current_time

            # 回调更新图像显示
            update_callback(processed_frame)

            # 控制处理速度，避免占用过多CPU
            time.sleep(0.01)

        # 释放资源
        self.video_capture.release()
        self.video_writer.release()

        # 如果视频被中断，删除未完成的输出文件
        if self.stop_event.is_set():
            self.video_interrupted = True
            if os.path.exists(output_video_path):
                os.remove(output_video_path)
            log_callback("[INFO] 视频检测已中断，未保存结果")
        else:
            log_callback(f"[INFO] 视频检测完成，结果已保存到: {output_video_path}")

    def stop_video_detection(self):
        """停止视频检测"""
        self.is_video_detection = False
        self.stop_event.set()
        if self.video_capture:
            self.video_capture.release()
        if self.video_writer:
            self.video_writer.release()

    def _extract_detection_info(self, results):
        """从检测结果中提取类别信息"""
        detections = {}
        for box in results.boxes:
            conf = float(box.conf[0])
            if conf < self.confidence_threshold:
                continue
            cls = int(box.cls[0])
            detections[cls] = detections.get(cls, 0) + 1
        return detections

    def draw_detections(self, img_path, results):
        """在图像上绘制检测结果并保存"""
        im = cv2.imread(img_path)
        if im is None:
            return None

        processed_im = self.draw_detections_on_frame(im, results)

        # 生成输出文件名
        input_path = Path(img_path)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"{input_path.stem}_detected_{timestamp}{input_path.suffix}"
        output_path = os.path.join(self.output_dir, output_filename)

        # 保存结果
        success = cv2.imwrite(output_path, processed_im)
        return output_path if success else None

    def draw_detections_on_frame(self, frame, results):
        """在帧上绘制检测结果"""
        im = frame.copy()
        height, width = im.shape[:2]

        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf = float(box.conf[0])
            cls = int(box.cls[0])

            if conf < self.confidence_threshold:
                continue

            # 绘制边界框
            color = (0, 255, 0)  # 绿色
            thickness = max(1, int(min(height, width) / 300))
            cv2.rectangle(im, (x1, y1), (x2, y2), color, thickness)

            # 绘制标签背景
            label = f"{COCO_NAMES[cls]}: {conf:.2f}"
            (label_width, label_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, thickness
            )

            cv2.rectangle(
                im,
                (x1, y1 - label_height - baseline - 5),
                (x1 + label_width, y1),
                color,
                -1
            )

            # 绘制标签文本
            cv2.putText(
                im,
                label,
                (x1, y1 - baseline - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                thickness
            )

        return im


# ---------- 主应用类 ----------
class ObjectDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("目标检测工具")
        self.root.geometry("1000x700")
        self.root.configure(bg="#f0f0f0")

        # 初始化变量
        self.image_path = None
        self.video_path = None
        self.detector = None
        self.output_path = None

        # 创建输出目录
        self.output_dir = self.setup_output_directory()

        # 修改：添加设备选择功能
        self.device = self.setup_device()

        # 设置样式
        self.setup_styles()

        # 创建界面
        self.create_widgets()

        # 日志重定向
        self.setup_logging()

    def setup_output_directory(self):
        """设置输出目录"""
        # 获取当前脚本所在目录
        current_dir = Path(__file__).parent if "__file__" in globals() else Path.cwd()
        output_dir = current_dir / "detection_results"

        # 创建目录（如果不存在）
        output_dir.mkdir(exist_ok=True)

        print(f"[INFO] 输出目录: {output_dir}")
        return output_dir

    def setup_device(self):
        """设置计算设备"""
        # 检查CUDA是否可用
        if torch.cuda.is_available():
            # 获取GPU数量
            gpu_count = torch.cuda.device_count()
            print(f"检测到 {gpu_count} 个GPU设备")

            # 显示每个GPU的信息
            for i in range(gpu_count):
                print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

            # 使用第一个GPU
            device = "cuda:0"
            print(f"使用设备: GPU - {torch.cuda.get_device_name(0)}")
        else:
            device = "cpu"
            print("使用设备: CPU")

        return device

    def setup_styles(self):
        """设置界面样式"""
        style = ttk.Style()
        style.theme_use('clam')

        # 配置样式
        style.configure('Title.TLabel',
                        font=('Arial', 16, 'bold'),
                        background='#f0f0f0',
                        foreground='#2c3e50')

        style.configure('Subtitle.TLabel',
                        font=('Arial', 12),
                        background='#f0f0f0',
                        foreground='#34495e')

        style.configure('TButton',
                        font=('Arial', 10),
                        padding=6)

        style.configure('Action.TButton',
                        font=('Arial', 10, 'bold'),
                        padding=8,
                        background='#3498db',
                        foreground='white')

        style.map('Action.TButton',
                  background=[('active', '#2980b9')])

        style.configure('Success.TButton',
                        background='#2ecc71',
                        foreground='white')

        style.map('Success.TButton',
                  background=[('active', '#27ae60')])

        style.configure('Warning.TButton',
                        background='#e74c3c',
                        foreground='white')

        style.map('Warning.TButton',
                  background=[('active', '#c0392b')])

    def setup_logging(self):
        """重定向日志到文本框"""

        class TextHandler:
            def __init__(self, text_widget):
                self.text_widget = text_widget

            def write(self, message):
                self.text_widget.insert(tk.END, message)
                self.text_widget.see(tk.END)

            def flush(self):
                pass

        sys.stdout = TextHandler(self.log_text)
        sys.stderr = TextHandler(self.log_text)

    def create_widgets(self):
        """创建界面组件"""
        # 主框架
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # 标题
        title_label = ttk.Label(main_frame, text="目标检测工具", style='Title.TLabel')
        title_label.grid(row=0, column=0, columnspan=2, pady=(0, 20))

        # 左侧面板 - 控制
        control_frame = ttk.LabelFrame(main_frame, text="控制面板", padding="10")
        control_frame.grid(row=1, column=0, sticky=(tk.N, tk.S, tk.W), padx=(0, 10))

        # 模型选择
        ttk.Label(control_frame, text="选择模型:", style='Subtitle.TLabel').grid(row=0, column=0, sticky=tk.W,
                                                                                 pady=(0, 5))

        self.model_var = tk.StringVar(value=list(MODEL_HUB.keys())[0])
        model_combo = ttk.Combobox(control_frame, textvariable=self.model_var,
                                   values=list(MODEL_HUB.keys()), state="readonly", width=20)
        model_combo.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 10))

        # 显示模型描述
        self.model_desc_var = tk.StringVar(value=MODEL_HUB[list(MODEL_HUB.keys())[0]][0])
        model_desc_label = ttk.Label(control_frame, textvariable=self.model_desc_var,
                                     style='Subtitle.TLabel', wraplength=200)
        model_desc_label.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=(0, 10))

        # 绑定模型选择事件
        model_combo.bind('<<ComboboxSelected>>', self.on_model_selected)

        # 置信度阈值
        ttk.Label(control_frame, text="置信度阈值:", style='Subtitle.TLabel').grid(row=3, column=0, sticky=tk.W,
                                                                                   pady=(10, 5))

        self.confidence_var = tk.DoubleVar(value=CONFIDENCE_THRESHOLD)
        confidence_scale = ttk.Scale(control_frame, from_=0.1, to=0.9, variable=self.confidence_var,
                                     orient=tk.HORIZONTAL)
        confidence_scale.grid(row=4, column=0, sticky=(tk.W, tk.E), pady=(0, 5))

        confidence_value = ttk.Label(control_frame, textvariable=self.confidence_var)
        confidence_value.grid(row=5, column=0, sticky=tk.W, pady=(0, 10))

        # 设备信息
        # 修改：显示更详细的设备信息
        device_text = f"GPU: {torch.cuda.get_device_name(0)}" if "cuda" in self.device else "CPU"
        ttk.Label(control_frame, text=f"计算设备: {device_text}", style='Subtitle.TLabel').grid(
            row=6, column=0, sticky=tk.W, pady=(10, 5))

        # 输出目录信息
        ttk.Label(control_frame, text=f"输出目录: {self.output_dir}", style='Subtitle.TLabel', wraplength=200).grid(
            row=7, column=0, sticky=tk.W, pady=(10, 5))

        # 模式选择
        ttk.Label(control_frame, text="检测模式:", style='Subtitle.TLabel').grid(row=8, column=0, sticky=tk.W,
                                                                                 pady=(10, 5))

        self.mode_var = tk.StringVar(value="image")
        mode_frame = ttk.Frame(control_frame)
        mode_frame.grid(row=9, column=0, sticky=(tk.W, tk.E), pady=(0, 10))

        ttk.Radiobutton(mode_frame, text="图像检测", variable=self.mode_var, value="image").pack(side=tk.LEFT)
        ttk.Radiobutton(mode_frame, text="视频检测", variable=self.mode_var, value="video").pack(side=tk.LEFT)

        # 按钮区域
        button_frame = ttk.Frame(control_frame)
        button_frame.grid(row=10, column=0, sticky=(tk.W, tk.E), pady=(20, 0))

        self.select_btn = ttk.Button(button_frame, text="选择文件", command=self.select_file, style='Action.TButton')
        self.select_btn.pack(side=tk.LEFT, padx=(0, 5))

        self.detect_btn = ttk.Button(button_frame, text="开始检测", command=self.start_detection,
                                     style='Action.TButton', state=tk.DISABLED)
        self.detect_btn.pack(side=tk.LEFT, padx=5)

        self.stop_btn = ttk.Button(button_frame, text="停止检测", command=self.stop_detection,
                                   style='Warning.TButton', state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=5)

        self.open_btn = ttk.Button(button_frame, text="打开结果", command=self.open_result,
                                   style='Success.TButton', state=tk.DISABLED)
        self.open_btn.pack(side=tk.LEFT, padx=(5, 0))

        # 右侧面板 - 图像显示
        image_frame = ttk.LabelFrame(main_frame, text="预览", padding="10")
        image_frame.grid(row=1, column=1, sticky=(tk.N, tk.S, tk.E, tk.W))

        self.canvas = tk.Canvas(image_frame, bg="white", width=600, height=400)
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # 底部面板 - 日志
        log_frame = ttk.LabelFrame(main_frame, text="日志输出", padding="10")
        log_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.S), pady=(10, 0))

        self.log_text = scrolledtext.ScrolledText(log_frame, height=8, width=80)
        self.log_text.pack(fill=tk.BOTH, expand=True)

        # 配置网格权重
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(1, weight=1)
        image_frame.rowconfigure(0, weight=1)
        image_frame.columnconfigure(0, weight=1)
        log_frame.columnconfigure(0, weight=1)

        # 状态栏
        self.status_var = tk.StringVar(value="就绪")
        status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.grid(row=1, column=0, sticky=(tk.W, tk.E))

        # 打印欢迎信息
        print(f"目标检测工具 - Ultralytics v{ultralytics_version}")
        print(f"使用设备: {device_text}")
        print(f"输出目录: {self.output_dir}")
        print("请选择文件并开始检测...")

    def on_model_selected(self, event):
        """模型选择事件处理"""
        selected_model = self.model_var.get()
        if selected_model in MODEL_HUB:
            self.model_desc_var.set(MODEL_HUB[selected_model][0])

    def select_file(self):
        """选择文件"""
        file_types = [
            ("图像文件", "*.jpg *.jpeg *.png *.bmp *.tiff"),
            ("视频文件", "*.mp4 *.avi *.mov *.mkv *.flv")
        ]

        file_path = filedialog.askopenfilename(title="选择文件", filetypes=file_types)

        if file_path:
            # 根据文件扩展名确定模式
            ext = Path(file_path).suffix.lower()
            if ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
                self.mode_var.set("image")
                self.image_path = file_path
                self.video_path = None
            else:
                self.mode_var.set("video")
                self.video_path = file_path
                self.image_path = None

            self.display_file(file_path)
            self.detect_btn.config(state=tk.NORMAL)
            self.status_var.set(f"已选择: {os.path.basename(file_path)}")
            print(f"[INFO] 已选择文件: {file_path}")

    def display_file(self, file_path):
        """显示文件"""
        try:
            # 根据文件类型选择显示方式
            ext = Path(file_path).suffix.lower()
            if ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
                # 显示图像
                img = Image.open(file_path)
                canvas_width = self.canvas.winfo_width()
                canvas_height = self.canvas.winfo_height()

                if canvas_width > 1 and canvas_height > 1:
                    img.thumbnail((canvas_width, canvas_height), Image.Resampling.LANCZOS)

                self.tk_image = ImageTk.PhotoImage(img)
                self.canvas.create_image(
                    canvas_width // 2,
                    canvas_height // 2,
                    image=self.tk_image,
                    anchor=tk.CENTER
                )
            else:
                # 显示视频第一帧
                cap = cv2.VideoCapture(file_path)
                ret, frame = cap.read()
                cap.release()

                if ret:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    img = Image.fromarray(frame_rgb)

                    canvas_width = self.canvas.winfo_width()
                    canvas_height = self.canvas.winfo_height()

                    if canvas_width > 1 and canvas_height > 1:
                        img.thumbnail((canvas_width, canvas_height), Image.Resampling.LANCZOS)

                    self.tk_image = ImageTk.PhotoImage(img)
                    self.canvas.create_image(
                        canvas_width // 2,
                        canvas_height // 2,
                        image=self.tk_image,
                        anchor=tk.CENTER
                    )
        except Exception as e:
            print(f"[ERROR] 无法显示文件: {e}")

    def update_video_frame(self, frame):
        """更新视频帧显示"""
        try:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)

            canvas_width = self.canvas.winfo_width()
            canvas_height = self.canvas.winfo_height()

            if canvas_width > 1 and canvas_height > 1:
                img.thumbnail((canvas_width, canvas_height), Image.Resampling.LANCZOS)

            self.tk_image = ImageTk.PhotoImage(img)
            self.canvas.delete("all")
            self.canvas.create_image(
                canvas_width // 2,
                canvas_height // 2,
                image=self.tk_image,
                anchor=tk.CENTER
            )
        except Exception as e:
            print(f"[ERROR] 更新视频帧失败: {e}")

    def start_detection(self):
        """开始目标检测"""
        mode = self.mode_var.get()

        if mode == "image" and not self.image_path:
            messagebox.showerror("错误", "请先选择图像!")
            return
        elif mode == "video" and not self.video_path:
            messagebox.showerror("错误", "请先选择视频!")
            return

        # 禁用按钮，防止重复点击
        self.select_btn.config(state=tk.DISABLED)
        self.detect_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL if mode == "video" else tk.DISABLED)
        self.status_var.set("正在加载模型...")

        # 在新线程中运行检测，防止界面卡死
        thread = threading.Thread(target=self.run_detection)
        thread.daemon = True
        thread.start()

    def run_detection(self):
        """运行目标检测"""
        try:
            model_key = self.model_var.get()
            model_name = MODEL_HUB[model_key][1]
            confidence = self.confidence_var.get()
            mode = self.mode_var.get()

            print(f"[INFO] 加载模型: {model_name} ...")
            print(f"[INFO] 使用设备: {self.device}")

            # 修改：传递设备参数和输出目录给Detector
            self.detector = Detector(model_name, self.device, confidence, self.output_dir)

            if mode == "image":
                print("[INFO] 正在进行图像检测...")
                results, output_path = self.detector.detect_image(self.image_path)
                self.output_path = output_path

                if output_path:
                    self.root.after(0, self.detection_completed)
                else:
                    self.root.after(0, self.detection_failed)

            else:  # video mode
                print("[INFO] 开始视频检测...")
                success = self.detector.detect_video(
                    self.video_path,
                    self.update_video_frame,
                    self.log_message
                )

                if not success:
                    self.root.after(0, self.detection_failed)

        except Exception as e:
            print(f"[ERROR] 检测过程中发生错误: {e}")
            import traceback
            traceback.print_exc()
            self.root.after(0, self.detection_failed)

    def log_message(self, message):
        """记录日志消息"""
        self.root.after(0, lambda: print(message))

    def stop_detection(self):
        """停止检测"""
        if self.detector:
            self.detector.stop_video_detection()
            self.status_var.set("检测已停止")
            self.select_btn.config(state=tk.NORMAL)
            self.detect_btn.config(state=tk.NORMAL)
            self.stop_btn.config(state=tk.DISABLED)

    def detection_completed(self):
        """检测完成后的回调"""
        self.select_btn.config(state=tk.NORMAL)
        self.detect_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self.open_btn.config(state=tk.NORMAL)
        self.status_var.set("检测完成!")

        # 显示结果图像
        if self.output_path and os.path.exists(self.output_path):
            self.display_file(str(self.output_path))

    def detection_failed(self):
        """检测失败后的回调"""
        self.select_btn.config(state=tk.NORMAL)
        self.detect_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self.status_var.set("检测失败!")

    def open_result(self):
        """打开结果文件或目录"""
        if self.output_path and os.path.exists(self.output_path):
            try:
                if sys.platform == 'win32':
                    os.startfile(self.output_path)
                elif sys.platform == 'darwin':  # macOS
                    os.system(f'open "{self.output_path}"')
                else:  # Linux
                    os.system(f'xdg-open "{self.output_path}"')
            except Exception as e:
                print(f"[ERROR] 无法打开文件: {e}")
                messagebox.showerror("错误", f"无法打开文件: {e}")
        else:
            # 如果没有具体文件路径，打开输出目录
            try:
                if sys.platform == 'win32':
                    os.startfile(self.output_dir)
                elif sys.platform == 'darwin':  # macOS
                    os.system(f'open "{self.output_dir}"')
                else:  # Linux
                    os.system(f'xdg-open "{self.output_dir}"')
            except Exception as e:
                print(f"[ERROR] 无法打开目录: {e}")
                messagebox.showerror("错误", f"无法打开目录: {e}")


# ---------- 主入口 ----------
if __name__ == '__main__':
    root = tk.Tk()
    app = ObjectDetectionApp(root)
    root.mainloop()