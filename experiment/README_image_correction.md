# 图像校正脚本使用说明

## 功能概述

此脚本用于校正SEM图像和普通手机拍摄图像的亮度、白平衡等问题，并按照原始目录结构保存到新文件夹中。

## 主要功能

### 普通图像校正 (image.jpg)
- **白平衡校正**: 使用Gray World算法校正色彩偏差
- **对比度增强**: 使用CLAHE (对比度限制自适应直方图均衡化)
- **亮度调整**: 通过Gamma校正优化整体亮度

### SEM图像校正 (sem.jpg)  
- **对比度增强**: 使用CLAHE提升细节对比度
- **直方图均衡化**: 改善整体亮度分布
- **降噪处理**: 使用双边滤波器减少噪声
- **细节保持**: 组合多种技术保持原始细节

### 文件管理
- **保留文件**: 
  - `image.jpg` (校正后的普通图像)
  - `sem.jpg` (校正后的SEM图像)
  - `sem.txt` (SEM元数据文件)
  - `C.jpeg, O.jpeg, Na.jpeg, Mg.jpeg, S.jpeg, Cl.jpeg, Ca.jpeg` (7种元素的原始EDS图像)
- **不保留文件**: 所有`*_processed.jpeg`文件
- **目录结构**: 完全保持原始目录结构

## 安装依赖

```bash
pip install -r requirements.txt
```

## 使用方法

### 基本用法
```bash
python image_correction.py
```
默认从`data`目录读取，输出到`data_corrected`目录

### 自定义路径
```bash
python image_correction.py --input /path/to/original/data --output /path/to/corrected/data
```

### 命令行参数
- `--input` 或 `-i`: 指定输入目录 (默认: `data`)
- `--output` 或 `-o`: 指定输出目录 (默认: `data_corrected`)

## 输出结构示例

```
data_corrected/
├── Table_1/
│   ├── Sample_1/
│   │   ├── 1_rep/
│   │   │   ├── image.jpg (校正后)
│   │   │   ├── sem.jpg (校正后)
│   │   │   ├── sem.txt (复制)
│   │   │   ├── C.jpeg (复制)
│   │   │   ├── O.jpeg (复制)
│   │   │   ├── Na.jpeg (复制)
│   │   │   ├── Mg.jpeg (复制)
│   │   │   ├── S.jpeg (复制)
│   │   │   ├── Cl.jpeg (复制)
│   │   │   └── Ca.jpeg (复制)
│   │   └── 2_rep/ to 5_rep/ (相同结构)
│   └── Sample_2/ to Sample_25/ (相同结构)
├── Table_2/ to Table_5/ (相同结构)
```

## 处理进度

脚本会显示处理进度，包括：
- 当前处理的表格
- 正在校正的图像文件
- 正在复制的文件

## 注意事项

1. 确保输入目录包含正确的数据结构
2. 输出目录会自动创建，如果已存在会覆盖同名文件
3. 处理大量数据时需要足够的磁盘空间
4. 建议在处理前备份原始数据

## 技术细节

### 白平衡校正算法
- 使用LAB色彩空间进行处理
- Gray World假设：平均灰度应为中性灰
- 调整a和b通道向中性点(128)靠拢

### SEM图像增强算法
- CLAHE: 局部对比度增强，防止过度增强
- 直方图均衡化: 改善整体亮度分布
- 加权组合: 保持原始细节的同时增强对比度
- 双边滤波: 降噪的同时保持边缘锐度 