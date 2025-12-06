#!/usr/bin/env python3
"""
主入口脚本 - 协调数据收集和可视化
"""

import argparse
import sys
import os

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_collection import DataCollector
from visualization import ResultVisualizer


def main():
    parser = argparse.ArgumentParser(description='目标检测模型基准测试工具')
    parser.add_argument('--mode', choices=['collect', 'visualize', 'all'],
                        default='all', help='运行模式: collect(仅收集数据), visualize(仅可视化), all(全部)')
    parser.add_argument('--results-dir', default='benchmark_results',
                        help='结果保存目录')

    args = parser.parse_args()

    print("=" * 60)
    print("目标检测模型基准测试工具")
    print("=" * 60)
    # ===== 新增代码开始 =====

    if args.mode in ['collect', 'all']:
        print("\n阶段1: 数据收集")
        print("-" * 30)
        collector = DataCollector()
        success = collector.run_benchmarks()

        if not success and args.mode == 'collect':
            return

    if args.mode in ['visualize', 'all']:
        print("\n阶段2: 结果可视化")
        print("-" * 30)
        try:
            visualizer = ResultVisualizer(args.results_dir)
            visualizer.load_results()
            visualizer.create_all_visualizations()
            visualizer.generate_report()
            print("✓ 可视化完成")
        except Exception as e:
            print(f"可视化失败: {e}")
            if args.mode == 'visualize':
                print("请先运行数据收集阶段")

    print("\n" + "=" * 60)
    print("基准测试完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()