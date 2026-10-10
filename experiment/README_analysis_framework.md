# 结晶数据集综合分析框架

## 概述

这是一个全面的多模态结晶数据集分析框架，包含统计分析、计算机视觉分析和机器学习分析。该数据集包含625个实验条件，系统地变化化学成分，提供三种不同的数据模态。

## 数据集结构

### 实验设计
- **5个实验表**: 不同的NaHCO₃浓度 (0.25, 0.5, 0.75, 1.0, 2.0 mM)
- **每表25个样本**: 5×5网格的CaCl₂ vs MgSO₄浓度组合
- **每个条件5次重复**: 总共625个实验样本
- **7种元素分析**: Ca, C, Cl, Mg, Na, O, S

### 数据模态
1. **化学成分数据**: 元素浓度矩阵
2. **扫描电子显微镜图像**: 高分辨率结构分析
3. **元素特异性映射图像**: 7种元素的空间分布信息

## 分析脚本

### 1. 统计分析 (`statistical_analysis.py`)

**功能**:
- 描述性统计分析
- 相关性分析 (Pearson和Spearman)
- 主成分分析 (PCA)
- 方差分析 (ANOVA/Kruskal-Wallis)

**数学方法**:
- Pearson相关: r = Σ[(xi - x̄)(yi - ȳ)] / √[Σ(xi - x̄)²Σ(yi - ȳ)²]
- 主成分分析: X = UΣV^T
- Kruskal-Wallis检验: H = (12/N(N+1)) Σ(Ri²/ni) - 3(N+1)

**输出文件**:
- statistical_summary.csv
- correlation_analysis.png
- pca_analysis.png
- descriptive_statistics.png

### 2. 计算机视觉分析 (`computer_vision_analysis.py`)

**功能**:
- 图像特征提取 (基础统计、纹理、形态学)
- 灰度共生矩阵 (GLCM) 纹理分析
- 形态学特征提取
- 边缘检测分析

**数学方法**:
- 灰度共生矩阵: P(i,j|δ,θ)
- 纹理特征对比度: Σi,j|i-j|²P(i,j)
- Shannon熵: H = -Σi P(i)log₂P(i)
- 偏心率: e = √(1 - (b²/a²))

**输出文件**:
- computer_vision_features.csv
- computer_vision_analysis.png
- cv_feature_summary.csv

### 3. 机器学习分析 (`machine_learning_analysis.py`)

**功能**:
- 分类任务 (表格预测、浓度分类)
- 回归任务 (预测实验条件)
- 特征重要性分析
- 多种算法比较 (RF, SVM, NN, 线性模型)

**数学方法**:
- 随机森林: ŷ = (1/B)ΣTb(x)
- 支持向量机: f(x) = Σαiyi K(xi,x) + b
- 神经网络: y = f(Σwixi + b)
- 交叉验证: CV = (1/k)Σi=1^k L(yi, ŷi)

**输出文件**:
- classification_results.csv
- regression_results.csv
- feature_importance.csv
- classification_results.png
- regression_results.png
- feature_importance.png

### 4. 综合分析 (`comprehensive_analysis.py`)

主脚本，运行所有分析并生成综合报告。

## 使用说明

### 安装依赖
```bash
pip install -r requirements.txt
```

### 运行分析

#### 单独运行各分析模块:
```bash
# 统计分析
python statistical_analysis.py

# 计算机视觉分析
python computer_vision_analysis.py

# 机器学习分析
python machine_learning_analysis.py
```

#### 运行完整分析:
```bash
python comprehensive_analysis.py
```

## 数据目录结构

```
experiment/
├── data/
│   ├── recipe.csv
│   ├── Table_1/
│   │   ├── Sample_1/
│   │   │   ├── 1_rep/
│   │   │   │   ├── sem.jpg          # SEM图像
│   │   │   │   ├── sem.txt          # SEM元数据
│   │   │   │   ├── image.jpg        # 光学显微镜图像
│   │   │   │   ├── C.jpeg           # 碳元素映射
│   │   │   │   ├── Ca.jpeg          # 钙元素映射
│   │   │   │   └── ...              # 其他元素映射
│   │   │   └── ...                  # 其他重复
│   │   └── ...                      # 其他样本
│   └── ...                          # 其他表格
├── recipies/
│   ├── Calcium.csv                  # 钙浓度矩阵
│   ├── Carbon.csv                   # 碳浓度矩阵
│   └── ...                          # 其他元素浓度矩阵
├── statistical_analysis.py         # 统计分析脚本
├── computer_vision_analysis.py     # 计算机视觉分析脚本
├── machine_learning_analysis.py    # 机器学习分析脚本
├── comprehensive_analysis.py       # 主分析脚本
├── requirements.txt                 # Python依赖包
└── README_analysis_framework.md    # 本文件
```

## 应用领域

### 1. 化学与材料科学
- 理解结晶机制
- 优化合成条件
- 预测晶体性质
- 研究元素分布模式

### 2. 机器学习
- 多模态学习算法开发
- 特征选择和工程方法
- 分类和回归算法比较
- 深度学习方法应用

### 3. 计算机视觉
- 纹理分析方法
- 形态学图像分析技术
- 科学图像处理算法
- 自动化材料表征系统

## 技术特点

- **多模态数据融合**: 结合化学、形态学和空间分布信息
- **系统性实验设计**: 受控变量的全面覆盖
- **高质量成像数据**: 多种成像模态，一致的采集参数
- **comprehensive分析框架**: 统计、计算机视觉和机器学习的完整工具链
- **reproducible研究**: 所有代码和数据完全开放，详细文档

## 输出结果解释

### 统计分析结果
- **相关性矩阵**: 显示元素间的线性和非线性关系
- **主成分分析**: 识别数据中的主要变异源
- **描述性统计**: 每种元素的分布特征

### 计算机视觉结果
- **纹理特征**: GLCM-based纹理量化
- **形态学特征**: 对象几何特性
- **空间分布**: 元素映射的空间模式

### 机器学习结果
- **分类性能**: 预测实验条件的准确性
- **回归性能**: 预测化学参数的精度
- **特征重要性**: 最具预测性的特征识别

## 引用和致谢

如果在研究中使用此数据集，请引用相关论文。该框架为结晶科学、预测材料设计和计算分析方法的研究提供了宝贵资源。

## 联系信息

如有问题或建议，请联系项目维护者。

---
*该分析框架提供了全面的多模态结晶数据分析工具，为化学、材料科学、机器学习和计算机视觉研究提供了强大的基础。* 