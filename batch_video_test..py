#!/usr/bin/env python3
"""
批量视频推理：11 模型 × 2 视频 → 画框视频 + FPS/耗时 csv（永久目录）
"""
import os
import time
import datetime
import cv2
import pandas as pd
from pathlib import Path
from ultralytics import YOLO

# ---------- 1. 模型清单 ----------
MODELS = {
    "yolov8n": "yolov8n.pt",
    "yolov8s": "yolov8s.pt",
    "yolov8m": "yolov8m.pt",
    "rtdetr-l": "rtdetr-l.pt",
    "rtdetr-x": "rtdetr-x.pt",
    "yolo11n": "yolo11n.pt",
    "yolo11m": "yolo11m.pt",
    "yolo11x": "yolo11x.pt",
    "yolo12n": "yolo12n.pt",
    "yolo12s": "yolo12s.pt",
    "yolo12m": "yolo12m.pt",
}

# ---------- 2. 视频路径 ----------
VIDEOS = {
    "road": r"D:\python xiangmu\ultralytics-main\pridect\video\道路上车辆行驶4K航拍.mp4",
    "sunset": r"D:\python xiangmu\ultralytics-main\pridect\video\行车记录仪：今天不拍路况，改拍夕阳.mp4",
}

# ---------- 3. 永久保存目录 ----------
SAVE_DIR = Path(r"D:\python xiangmu\ultralytics-main\pridect\shiyan\runs\video_benchmark")
SAVE_DIR.mkdir(parents=True, exist_ok=True)
CSV_PATH = SAVE_DIR / "video_benchmark_results.csv"


def process_video(model, video_path, output_path):
    """手动处理视频并保存到指定路径"""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  错误：无法打开视频 {video_path}")
        return 0

    # 获取视频属性
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 创建视频写入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    frame_count = 0
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 推理
        results = model(frame, verbose=False)

        # 绘制检测结果
        annotated_frame = results[0].plot()

        # 写入帧
        out.write(annotated_frame)
        frame_count += 1

        # 进度显示
        if frame_count % 100 == 0:
            elapsed = time.time() - start_time
            print(f"    已处理 {frame_count}/{total_frames} 帧, 耗时: {elapsed:.1f}s")

    # 释放资源
    cap.release()
    out.release()
    processing_time = time.time() - start_time

    return frame_count, processing_time


def main():
    records = []

    for model_name, weight in MODELS.items():
        print(f"\n>>> 加载模型：{model_name}")
        try:
            model = YOLO(weight)
        except Exception as e:
            print(f"  模型加载失败: {e}")
            continue

        for video_key, video_path in VIDEOS.items():
            vp = Path(video_path)
            if not vp.exists():
                print(f"  跳过：{video_path} 不存在")
                continue

            # 定义输出路径
            out_path = SAVE_DIR / f"{model_name}_{video_key}.mp4"

            # 如果文件已存在，询问是否覆盖
            if out_path.exists():
                response = input(f"  文件 {out_path.name} 已存在，是否覆盖？(y/n): ")
                if response.lower() != 'y':
                    print(f"  跳过 {out_path.name}")
                    continue

            print(f"  处理 {video_key} -> {out_path}")

            # 处理视频
            frame_count, processing_time = process_video(model, video_path, out_path)

            if frame_count > 0:
                fps = frame_count / processing_time if processing_time > 0 else 0

                print(f"✅ 完成: {frame_count} 帧, 耗时: {processing_time:.1f}s, FPS: {fps:.1f}")

                records.append({
                    "模型": model_name,
                    "视频": video_key,
                    "帧数": frame_count,
                    "耗时(s)": round(processing_time, 2),
                    "FPS": round(fps, 1),
                    "结果文件": out_path.name,
                    "时间戳": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                })
            else:
                print(f"❌ 处理失败: {video_key}")

    # 保存结果到CSV
    if records:
        df = pd.DataFrame(records)

        # 如果CSV已存在，追加数据
        if CSV_PATH.exists():
            existing_df = pd.read_csv(CSV_PATH)
            df = pd.concat([existing_df, df], ignore_index=True)

        df.to_csv(CSV_PATH, index=False, encoding="utf-8-sig")
        print(f"\n✅ 完成！结果已保存到: {CSV_PATH}")
        print("\n最新记录:")
        print(pd.DataFrame(records).to_string(index=False))
    else:
        print("\n❌ 没有生成任何记录")


if __name__ == "__main__":
    main()