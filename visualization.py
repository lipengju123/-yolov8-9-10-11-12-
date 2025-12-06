#!/usr/bin/env python3
"""
可视化模块 - 负责读取数据并生成各种图表
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate
import json
import warnings

from config import OUTPUT_CONFIG, VISUALIZATION_CONFIG

# 设置matplotlib后端和字体
plt.switch_backend('Agg')
plt.rcParams['font.family'] = VISUALIZATION_CONFIG["font_family"]
plt.rcParams['axes.unicode_minus'] = False
warnings.filterwarnings("ignore")


class ResultVisualizer:
    def __init__(self, results_dir=None):
        if results_dir is None:
            results_dir = OUTPUT_CONFIG["base_dir"]

        self.base_dir = results_dir
        self.data_dir = os.path.join(results_dir, OUTPUT_CONFIG["data_dir"])
        self.plots_dir = os.path.join(results_dir, OUTPUT_CONFIG["plots_dir"])

        # 检查目录是否存在
        if not os.path.exists(self.data_dir):
            raise FileNotFoundError(f"数据目录不存在: {self.data_dir}")

        self.df = None
        self.results_data = None
        import matplotlib.pyplot as plt
        from matplotlib import font_manager

        # 1. 找到系统里 100% 有中文的字体
        chinese_font = font_manager.FontProperties(
            fname=font_manager.findfont("SimHei")  # Windows 黑体
            # fname="/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"  # Linux 例子
            # fname="/System/Library/Fonts/PingFang.ttc"  # macOS 例子
        )

        # 2. 全局生效
        plt.rcParams["font.family"] = chinese_font.get_name()
        plt.rcParams["axes.unicode_minus"] = False  # 负号正常显示

    def load_results(self):
        """加载基准测试结果"""
        csv_path = os.path.join(self.data_dir, "benchmark_results.csv")
        json_path = os.path.join(self.data_dir, "benchmark_results.json")

        if os.path.exists(csv_path):
            self.df = pd.read_csv(csv_path)
            print(f"✓ 从CSV加载结果: {len(self.df)} 个模型")
        elif os.path.exists(json_path):
            with open(json_path, 'r', encoding='utf-8') as f:
                self.results_data = json.load(f)
            self.df = pd.DataFrame(self.results_data["results"])
            print(f"✓ 从JSON加载结果: {len(self.df)} 个模型")
        else:
            raise FileNotFoundError(f"在 {self.data_dir} 中找不到结果文件")

        return self.df

    def create_accuracy_comparison(self):
        """创建精度对比图"""
        if self.df is None:
            self.load_results()

        fig, ax = plt.subplots(figsize=VISUALIZATION_CONFIG["figsize"])

        models = self.df["模型名称"]
        map50_95 = self.df["mAP50-95"]
        map50 = self.df["mAP50"]
        map75 = self.df["mAP75"]

        x = np.arange(len(models))
        width = 0.25

        ax.bar(x - width, map50_95, width, label='mAP50-95', color='steelblue')
        ax.bar(x, map50, width, label='mAP50', color='lightcoral')
        ax.bar(x + width, map75, width, label='mAP75', color='mediumseagreen')

        ax.set_xlabel('模型')
        ax.set_ylabel('精度')
        ax.set_title('模型精度对比')
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f'{self.plots_dir}/01_accuracy_comparison.png',
                    dpi=VISUALIZATION_CONFIG["dpi"], bbox_inches='tight')
        plt.close()
        print("✓ 创建精度对比图")

    def create_speed_comparison(self):
        """创建速度对比图"""
        if self.df is None:
            self.load_results()

        fig, ax = plt.subplots(figsize=VISUALIZATION_CONFIG["figsize"])

        models = self.df["模型名称"]
        fps = self.df["推理速度(FPS)"]

        bars = ax.bar(models, fps, color='orange', alpha=0.7)

        for bar, value in zip(bars, fps):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    f'{value:.1f}', ha='center', va='bottom', fontsize=12)

        ax.set_xlabel('模型')
        ax.set_ylabel('FPS')
        ax.set_title('推理速度对比 (越高越好)')
        ax.set_xticklabels(models, rotation=45)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f'{self.plots_dir}/02_speed_comparison.png',
                    dpi=VISUALIZATION_CONFIG["dpi"], bbox_inches='tight')
        plt.close()
        print("✓ 创建速度对比图")

    def create_training_curves(self):
        """画训练过程曲线：loss + precision + recall + mAP"""
        if self.df is None:
            self.load_results()

        for model_name in self.df["模型文件"].str.replace(".pt", ""):
            csv_path = os.path.join(self.base_dir, OUTPUT_CONFIG["detailed_dir"],
                                    f"{model_name}_results.csv")
            if not os.path.exists(csv_path):
                continue

            r = pd.read_csv(csv_path)

            fig, ax = plt.subplots(2, 2, figsize=(12, 10))
            fig.suptitle(f"{model_name} 训练曲线", fontsize=16)

            # 1. loss
            ax[0, 0].plot(r["epoch"], r["train/box_loss"], label="train/box_loss")
            ax[0, 0].plot(r["epoch"], r["val/box_loss"], label="val/box_loss")
            ax[0, 0].set_title("Box Loss")
            ax[0, 0].legend()

            # 2. cls_loss
            ax[0, 1].plot(r["epoch"], r["train/cls_loss"], label="train/cls_loss")
            ax[0, 1].plot(r["epoch"], r["val/cls_loss"], label="val/cls_loss")
            ax[0, 1].set_title("Cls Loss")
            ax[0, 1].legend()

            # 3. precision/recall
            ax[1, 0].plot(r["epoch"], r["metrics/precision(B)"], label="precision")
            ax[1, 0].plot(r["epoch"], r["metrics/recall(B)"], label="recall")
            ax[1, 0].set_title("Precision & Recall")
            ax[1, 0].legend()

            # 4. mAP
            ax[1, 1].plot(r["epoch"], r["metrics/mAP50(B)"], label="mAP50")
            ax[1, 1].plot(r["epoch"], r["metrics/mAP50-95(B)"], label="mAP50-95")
            ax[1, 1].set_title("mAP")
            ax[1, 1].legend()

            plt.tight_layout()
            out_path = os.path.join(self.plots_dir, f"{model_name}_training_curves.png")
            plt.savefig(out_path, dpi=VISUALIZATION_CONFIG["dpi"], bbox_inches='tight')
            plt.close()
            print(f"✓ 训练曲线已保存: {out_path}")
    def create_speed_accuracy_tradeoff(self):
        """创建精度-速度权衡图"""
        if self.df is None:
            self.load_results()

        fig, ax = plt.subplots(figsize=VISUALIZATION_CONFIG["figsize"])

        fps = self.df["推理速度(FPS)"]
        map50_95 = self.df["mAP50-95"]
        models = self.df["模型名称"]

        ax.scatter(fps, map50_95, s=150, c='red', alpha=0.7)

        for i, model in enumerate(models):
            ax.annotate(model, (fps[i], map50_95[i]), xytext=(5, 5),
                        textcoords='offset points', fontsize=11)

        ax.set_xlabel('推理速度 (FPS)')
        ax.set_ylabel('mAP50-95')
        ax.set_title('精度-速度权衡')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f'{self.plots_dir}/03_speed_accuracy_tradeoff.png',
                    dpi=VISUALIZATION_CONFIG["dpi"], bbox_inches='tight')
        plt.close()
        print("✓ 创建精度-速度权衡图")

    def create_model_size_comparison(self):
        """创建模型大小对比图"""
        if self.df is None:
            self.load_results()

        fig, ax = plt.subplots(figsize=VISUALIZATION_CONFIG["figsize"])

        models = self.df["模型名称"]
        model_sizes = self.df["模型大小(MB)"]

        bars = ax.bar(models, model_sizes, color='green', alpha=0.7)

        for bar, value in zip(bars, model_sizes):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    f'{value:.1f}', ha='center', va='bottom', fontsize=12)

        ax.set_xlabel('模型')
        ax.set_ylabel('模型大小 (MB)')
        ax.set_title('模型文件大小对比')
        ax.set_xticklabels(models, rotation=45)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f'{self.plots_dir}/04_model_size_comparison.png',
                    dpi=VISUALIZATION_CONFIG["dpi"], bbox_inches='tight')
        plt.close()
        print("✓ 创建模型大小对比图")

    def create_efficiency_comparison(self):
        """创建效率对比图"""
        if self.df is None:
            self.load_results()

        fig, ax = plt.subplots(figsize=VISUALIZATION_CONFIG["figsize"])

        models = self.df["模型名称"]
        efficiency_scores = self.df["mAP50-95"] * self.df["推理速度(FPS)"] / self.df["模型大小(MB)"]

        bars = ax.bar(models, efficiency_scores, color='purple', alpha=0.7)

        for bar, value in zip(bars, efficiency_scores):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f'{value:.2f}', ha='center', va='bottom', fontsize=12)

        ax.set_xlabel('模型')
        ax.set_ylabel('效率得分')
        ax.set_title('综合效率对比 (mAP×FPS/模型大小)')
        ax.set_xticklabels(models, rotation=45)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f'{self.plots_dir}/05_efficiency_comparison.png',
                    dpi=VISUALIZATION_CONFIG["dpi"], bbox_inches='tight')
        plt.close()
        print("✓ 创建效率对比图")

    def create_all_visualizations(self):
        """创建所有可视化图表"""
        if self.df is None:
            self.load_results()

        print("开始创建可视化图表...")

        # 创建所有图表
        self.create_accuracy_comparison()
        self.create_speed_comparison()
        self.create_speed_accuracy_tradeoff()
        self.create_model_size_comparison()
        self.create_efficiency_comparison()
        self.create_efficiency_comparison()
        # 新增
        self.create_training_curves()
        print(f"\n所有图表已保存到: {self.plots_dir}")

    def generate_report(self):
        """生成文本报告"""
        if self.df is None:
            self.load_results()

        report_path = os.path.join(self.base_dir, "performance_report.md")

        with open(report_path, "w", encoding="utf-8") as f:
            f.write("# 目标检测模型性能对比报告\n\n")

            f.write("## 性能对比结果\n\n")
            f.write(tabulate(self.df, headers='keys', tablefmt='github', showindex=False))
            f.write("\n\n")

            # 找出最佳模型
            best_map_model = self.df.loc[self.df['mAP50-95'].idxmax()]['模型名称']
            best_speed_model = self.df.loc[self.df['推理速度(FPS)'].idxmax()]['模型名称']

            f.write("## 关键发现\n\n")
            f.write(f"- **精度最高**: {best_map_model} (mAP50-95: {self.df['mAP50-95'].max():.4f})\n")
            f.write(f"- **速度最快**: {best_speed_model} ({self.df['推理速度(FPS)'].max():.2f} FPS)\n")

            # 计算效率得分
            self.df['效率得分'] = self.df['mAP50-95'] * self.df['推理速度(FPS)'] / self.df['模型大小(MB)']
            best_efficiency_model = self.df.loc[self.df['效率得分'].idxmax()]['模型名称']
            f.write(f"- **效率最佳**: {best_efficiency_model}\n\n")

            f.write("## 可视化图表\n\n")
            f.write("1. **精度对比图** - 比较不同模型的mAP指标\n")
            f.write("2. **速度对比图** - 比较推理速度(FPS)\n")
            f.write("3. **精度-速度权衡图** - 展示精度和速度的关系\n")
            f.write("4. **模型大小对比图** - 比较模型文件大小\n")
            f.write("5. **效率对比图** - 综合考量精度、速度和模型大小\n")

        print(f"✓ 报告已生成: {report_path}")


def main():
    """可视化主函数"""
    try:
        visualizer = ResultVisualizer()
        visualizer.load_results()

        # 打印结果表格
        print("\n" + "=" * 60)
        print("模型性能对比结果:")
        print("=" * 60)
        print(tabulate(visualizer.df, headers='keys', tablefmt='grid', showindex=False))

        # 生成可视化图表
        visualizer.create_all_visualizations()

        # 生成报告
        visualizer.generate_report()

        print(f"\n可视化完成！结果保存在: {visualizer.base_dir}")

    except Exception as e:
        print(f"错误: {e}")
        print("请先运行 data_collection.py 收集数据")


if __name__ == "__main__":
    main()