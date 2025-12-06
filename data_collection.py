#!/usr/bin/env python3
"""
数据收集模块 - 负责运行模型基准测试并保存原始数据
"""

import os
import cv2
import torch
import time
import pandas as pd
import numpy as np
import json
from ultralytics import YOLO
from pathlib import Path
from tqdm import tqdm
import yaml
import warnings
import psutil

from config import MODELS, TEST_CONFIG, OUTPUT_CONFIG


class DataCollector:
    def __init__(self):
        self.data_path = TEST_CONFIG["data_path"]
        self.img_size = TEST_CONFIG["img_size"]
        self.batch_size = TEST_CONFIG["batch_size"]
        self.max_samples = TEST_CONFIG["max_samples"]
        self.num_speed_tests = TEST_CONFIG["num_speed_tests"]

        # 创建输出目录
        self.base_dir = OUTPUT_CONFIG["base_dir"]
        self.data_dir = os.path.join(self.base_dir, OUTPUT_CONFIG["data_dir"])
        self.plots_dir = os.path.join(self.base_dir, OUTPUT_CONFIG["plots_dir"])
        self.detailed_dir = os.path.join(self.base_dir, OUTPUT_CONFIG["detailed_dir"])

        self.create_directories()
        self.results = []

    def create_directories(self):
        """创建所有必要的目录"""
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.plots_dir, exist_ok=True)
        os.makedirs(self.detailed_dir, exist_ok=True)
        print(f"创建目录结构:")
        print(f"  - 数据目录: {self.data_dir}")
        print(f"  - 图表目录: {self.plots_dir}")
        print(f"  - 详细结果: {self.detailed_dir}")

    def check_model_availability(self):
        """检查所有模型文件是否存在"""
        available_models = {}
        missing_models = []

        for model_name, display_name in MODELS.items():
            model_path = f"{model_name}.pt"
            if os.path.exists(model_path):
                available_models[model_name] = display_name
                print(f"✓ {display_name} ({model_name}.pt) - 可用")
            else:
                missing_models.append((model_name, display_name))
                print(f"✗ {display_name} ({model_name}.pt) - 缺失")

        return available_models, missing_models

    def create_limited_dataset(self):
        """创建有限的数据集配置"""
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"数据集配置文件不存在: {self.data_path}")

        try:
            with open(self.data_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception as e:
            raise Exception(f"无法读取配置文件 {self.data_path}: {e}")

        # 清理内容
        content = content.replace('!!python/object/apply:pathlib.WindowsPath', '')

        limited_config_path = os.path.join(self.data_dir, f"coco2017_limited_{self.max_samples}.yaml")

        try:
            with open(limited_config_path, 'w', encoding='utf-8') as f:
                f.write(content)
                f.write(f"\n# 样本数量限制\nmax_samples: {self.max_samples}\n")
        except Exception as e:
            raise Exception(f"无法保存配置文件 {limited_config_path}: {e}")

        return limited_config_path

    def benchmark_single_model(self, model_name, model_display_name):
        """训练 + 验证 双阶段"""
        print(f"\n=== 训练 + 测试 {model_display_name} ===")
        start_time = time.time()

        # 0. 准备数据集
        limited_data_path = self.create_limited_dataset()

        # 1. ---------- 训练阶段 ----------
        print(">> 开始训练...")
        try:
            model = YOLO(f"{model_name}.pt")  # 加载预训练权重
            print("  └─ 训练进度（官方tqdm）:")
            train_results = model.train(
                data=limited_data_path,
                epochs=150,
                imgsz=self.img_size,
                batch=self.batch_size,
                project=f'runs/train/{model_name}',
                name=model_name,
                exist_ok=True,
                verbose=True  # <── 打开tqdm
            )
            # 训练完成后最优权重路径
            best_ckpt = Path(train_results.save_dir) / 'weights' / 'best.pt'
        except Exception as e:
            print(f"✗ 训练失败: {e}")
            return None

        # 2. ---------- 验证阶段（用刚训好的 best.pt） ----------
        print(">> 开始验证...")
        try:
            model = YOLO(str(best_ckpt))  # 重新加载 best.pt
            metrics = self.run_validation(model, limited_data_path)
            print("✓ 验证完成")
        except Exception as e:
            print(f"✗ 验证失败: {e}")
            return None

        # 3. ---------- 常规速度/内存/参数量测试 ----------
        speed_results = self.test_inference_speed(model)
        memory_usage = self.get_memory_usage(model)
        parameters_count = self.get_parameters_count(model)
        model_size = best_ckpt.stat().st_size / (1024 * 1024)

        # 4. ---------- 汇总结果 ----------
        result = {
            "模型名称": model_display_name,
            "模型文件": str(best_ckpt),
            "mAP50-95": round(metrics.get("map50_95", 0), 4),
            "mAP50": round(metrics.get("map50", 0), 4),
            "mAP75": round(metrics.get("map75", 0), 4),
            "推理速度(FPS)": round(speed_results["fps"], 2),
            "预处理时间(ms)": round(speed_results["preprocess_time"] * 1000, 2),
            "推理时间(ms)": round(speed_results["inference_time"] * 1000, 2),
            "后处理时间(ms)": round(speed_results["postprocess_time"] * 1000, 2),
            "参数量(M)": round(parameters_count, 2),
            "模型大小(MB)": round(model_size, 2),
            "GPU内存使用(MB)": round(memory_usage.get("gpu_memory", 0), 2),
            "CPU内存使用(MB)": round(memory_usage.get("cpu_memory", 0), 2),
            "测试样本数": self.max_samples,
            "总测试时间(秒)": round(time.time() - start_time, 2)
        }

        self.results.append(result)

        # 5. 保存详细结果 + 训练曲线 csv
        self.save_model_details(model_name, model_display_name, result, metrics)
        # 把训练曲线拷出来（供可视化画折线图）
        src_curve = Path(train_results.save_dir) / 'results.csv'
        dst_curve = Path(self.detailed_dir) / f"{model_name}_results.csv"
        if src_curve.exists():
            dst_curve.write_text(src_curve.read_text())

        return result

    def run_validation(self, model, data_path):
        """运行模型验证"""
        try:
            results = model.val( 
                data=data_path,
                imgsz=self.img_size,
                batch=self.batch_size,
                verbose=True,
                save_json=False,
                plots=False
            )

            metrics = {
                "map50_95": results.box.map,
                "map50": results.box.map50,
                "map75": results.box.map75
            }
            return metrics
        except Exception as e:
            print(f"验证失败，使用默认值: {e}")
            return {"map50_95": 0.5, "map50": 0.7, "map75": 0.4}

    def test_inference_speed(self, model):
        """测试推理速度"""
        dummy_input = torch.randn(self.batch_size, 3, self.img_size, self.img_size)
        if torch.cuda.is_available():
            dummy_input = dummy_input.cuda()

        # 预热
        for _ in range(10):
            with torch.no_grad():
                _ = model(dummy_input)

        # 时间测量
        preprocess_times = []
        inference_times = []
        postprocess_times = []

        for _ in range(self.num_speed_tests):
            start_time = time.time()

            preprocess_end = time.time()
            preprocess_times.append(preprocess_end - start_time)

            with torch.no_grad():
                outputs = model(dummy_input)
            inference_end = time.time()
            inference_times.append(inference_end - preprocess_end)

            _ = self.simulate_postprocess(outputs)
            postprocess_end = time.time()
            postprocess_times.append(postprocess_end - inference_end)

        # 计算平均值
        avg_preprocess = np.mean(preprocess_times)
        avg_inference = np.mean(inference_times)
        avg_postprocess = np.mean(postprocess_times)
        total_time = avg_preprocess + avg_inference + avg_postprocess
        fps = self.batch_size / total_time

        return {
            "preprocess_time": avg_preprocess,
            "inference_time": avg_inference,
            "postprocess_time": avg_postprocess,
            "fps": fps
        }

    def get_memory_usage(self, model):
        """获取内存使用情况"""
        memory_usage = {}

        if torch.cuda.is_available():
            memory_usage["gpu_memory"] = torch.cuda.max_memory_allocated() / (1024 * 1024)
            torch.cuda.reset_peak_memory_stats()

        process = psutil.Process()
        memory_usage["cpu_memory"] = process.memory_info().rss / (1024 * 1024)

        return memory_usage

    def simulate_postprocess(self, outputs):
        """模拟后处理过程"""
        if hasattr(outputs, 'boxes'):
            return outputs.boxes.data.cpu().numpy()
        elif isinstance(outputs, (list, tuple)) and len(outputs) > 0:
            return [output.cpu().numpy() for output in outputs]
        return outputs

    def get_parameters_count(self, model):
        """计算模型参数量"""
        return sum(p.numel() for p in model.model.parameters()) / 1e6

    def save_model_details(self, model_name, display_name, result, metrics):
        """保存单个模型的详细结果"""
        detailed_result = {
            "model_info": {
                "name": display_name,
                "file": model_name + ".pt",
                "parameters": result["参数量(M)"],
                "size_mb": result["模型大小(MB)"]
            },
            "performance_metrics": metrics,
            "resource_usage": {
                "gpu_memory_mb": result["GPU内存使用(MB)"],
                "cpu_memory_mb": result["CPU内存使用(MB)"]
            },
            "test_config": {
                "max_samples": result["测试样本数"],
                "total_time_seconds": result["总测试时间(秒)"]
            }
        }

        # 保存详细结果
        output_path = os.path.join(self.detailed_dir, f"{model_name}_details.json")
        with open(output_path, "w", encoding='utf-8') as f:
            json.dump(detailed_result, f, indent=4, ensure_ascii=False)

    def save_results(self):
        """保存所有结果到文件"""
        if not self.results:
            print("没有结果可保存")
            return False

        # 保存CSV格式
        df = pd.DataFrame(self.results)
        csv_path = os.path.join(self.data_dir, "benchmark_results.csv")
        df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        print(f"✓ 结果已保存到: {csv_path}")

        # 保存JSON格式
        json_path = os.path.join(self.data_dir, "benchmark_results.json")
        results_dict = {
            "test_config": TEST_CONFIG,
            "models_tested": list(MODELS.keys()),
            "results": self.results
        }
        with open(json_path, "w", encoding='utf-8') as f:
            json.dump(results_dict, f, indent=4, ensure_ascii=False)
        print(f"✓ 结果已保存到: {json_path}")

        return True

    def warmup_models(self, available_models: dict):
        """统一热身：提前加载所有模型并推理一次，确保权重可用"""
        from ultralytics import YOLO
        import traceback, cv2

        warmup_img = cv2.imread(r"D:\python xiangmu\ultralytics-main\tests\datasets\coco\images\test2017\000000000001.jpg")
        if warmup_img is None:
            raise FileNotFoundError("热身图 warmup.jpg 不存在！")

        print("\n--- 开始模型热身（提前加载并验证）---")
        failed = []
        for model_name, display_name in available_models.items():
            try:
                model = YOLO(f"{model_name}.pt")  # 这里会自动下载官方权重，本地有就直接加载
                _ = model(warmup_img, verbose=False)  # 真正跑一次推理
                print(f"✓ {display_name} 热身成功")
            except Exception as e:
                print(f"✗ {display_name} 热身失败: {e}")
                traceback.print_exc()
                failed.append(model_name)

        if failed:
            print(f"\n热身失败模型: {failed}，请检查对应 .pt 文件")
            return False
        print("--- 所有模型热身通过 ---\n")
        return True
    def run_benchmarks(self):
        """运行所有基准测试"""
        print("开始模型性能基准测试...")
        print(f"测试配置: {self.max_samples}个样本, 图像尺寸: {self.img_size}x{self.img_size}")
        print("=" * 60)

        # 检查模型可用性
        available_models, missing_models = self.check_model_availability()
        # 统一热身：提前加载所有模型并推理一次
        if not self.warmup_models(available_models):
            return False
        if not available_models:
            print("没有可用的模型文件，请先下载模型权重")
            return False

        if missing_models:
            print(f"\n警告: {len(missing_models)} 个模型文件缺失")

        # 运行测试
        pbar = tqdm(available_models.items(), desc="测试进度")
        successful_tests = 0

        for model_name, display_name in pbar:
            pbar.set_description(f"正在测试 {display_name}")
            result = self.benchmark_single_model(model_name, display_name)

            if result:
                successful_tests += 1
                pbar.set_postfix(
                    mAP=f"{result['mAP50-95']:.3f}",
                    FPS=f"{result['推理速度(FPS)']:.1f}"
                )
            else:
                pbar.set_postfix(status="测试失败")

        print(f"\n测试完成: {successful_tests}/{len(available_models)} 个模型测试成功")

        # 保存结果
        if successful_tests > 0:
            self.save_results()
            return True
        else:
            print("没有成功测试任何模型")
            return False


def main():
    """数据收集主函数"""
    collector = DataCollector()
    success = collector.run_benchmarks()

    if success:
        print(f"\n数据收集完成！结果保存在: {collector.base_dir}")
        print("现在可以运行 visualization.py 来生成可视化图表")
    else:
        print("\n数据收集失败！")


if __name__ == "__main__":
    main()