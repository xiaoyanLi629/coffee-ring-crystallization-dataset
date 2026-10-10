#!/usr/bin/env python3
"""
快速运行图像校正脚本
"""

import os
import sys
from pathlib import Path

def check_dependencies():
    """检查必要的依赖包"""
    try:
        import cv2
        import numpy as np
        print("✓ 所有依赖包已安装")
        return True
    except ImportError as e:
        print(f"✗ 缺少依赖包: {e}")
        print("请运行: pip install -r requirements.txt")
        return False

def main():
    print("=" * 60)
    print("        图像校正脚本 - 快速运行工具")
    print("=" * 60)
    
    # 检查依赖
    if not check_dependencies():
        return
    
    # 检查输入目录
    input_dir = input("请输入数据目录路径 (默认: data): ").strip()
    if not input_dir:
        input_dir = "data"
    
    if not Path(input_dir).exists():
        print(f"✗ 输入目录不存在: {input_dir}")
        return
    
    # 设置输出目录
    output_dir = input("请输入输出目录路径 (默认: data_corrected): ").strip()
    if not output_dir:
        output_dir = "data_corrected"
    
    # 确认操作
    print(f"\n配置信息:")
    print(f"  输入目录: {input_dir}")
    print(f"  输出目录: {output_dir}")
    
    confirm = input("\n确认开始处理? (y/N): ").strip().lower()
    if confirm not in ['y', 'yes', '是']:
        print("操作已取消")
        return
    
    # 导入并运行校正器
    try:
        from image_correction import ImageCorrector
        
        corrector = ImageCorrector(input_dir, output_dir)
        corrector.process_all()
        
        print("\n" + "=" * 60)
        print("✓ 图像校正完成!")
        print(f"✓ 校正后的图像已保存到: {output_dir}")
        print("=" * 60)
        
    except Exception as e:
        print(f"✗ 处理过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 