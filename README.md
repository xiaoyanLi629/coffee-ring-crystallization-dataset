# 咖啡环残留物多模态结晶数据集（625 样本）

这是数据集论文 *A Comprehensive Multi-Modal Crystallization Dataset for Machine Learning and Computer Vision Applications* 的独立项目目录。数据集共 125 个化学配方，每个配方重复 5 次，合计 625 个样本；配方由 NaHCO₃（5 个水平，对应 Table_1 到 Table_5）与 CaCl₂、MgSO₄ 的 5×5 组合构成。每个样本包含一张 SEM 图（sem.jpg 及仪器参数 sem.txt）、一张手机光学照片（image.jpg）和七张 EDS 元素图（C、Ca、Cl、Mg、Na、O、S）。

本目录于 2026-10-10 从建模项目 `CNN-Vision-Transformer_Model_for_Elements_Concentration_Estimation_by_Coffee-Ring_Effect_Residue_Patterns/dataset_paper/` 整体移出，内部结构保持原样，因此分析脚本里的相对路径不需要改。建模项目自己的 `data/` 保留了一份同样的原始数据（逐文件大小比对一致），两个项目各自独立，删改任何一边都不会影响另一边。

## 目录说明

`data/` 是数据集的发布版本，包含 Table_1 到 Table_5 的原始图像、配方与元素浓度表（recipe.csv、element_concentrations.csv、organized_recipes_detailed*.csv）以及描述性统计图（analysis/）。

`experiment/` 存原始数据：`experiment/data/` 是每个液滴的原始 SEM、EDS、光学图像和 sem.txt（另含 2021-08-31 生成的派生图 *_processed.jpeg），`recipies/` 是各元素浓度表与配方原始 xlsx。旧的统计、机器学习和图像校正脚本已于 2026-10-14 删除（在 git 历史里），`experiment/data_corrected/` 是旧校正脚本的输出，未入库。所有现行分析在 `analysis/`。

`manuscript/` 是投 Scientific Data 的 Data Descriptor 稿件：manuscript.tex（Springer Nature sn-jnl 模板）、references.bib、figures/ 和已编译的 manuscript.pdf。图 1、2 由 `analysis/ms_figs.py` 生成，其余图来自各分析模块。

## 状态

论文尚未投稿。图像数据不入 git（见 .gitignore），只跟踪代码、表格、结果和稿件。
