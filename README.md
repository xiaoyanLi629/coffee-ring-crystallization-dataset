# 咖啡环残留物多模态结晶数据集（625 样本）

这是 Scientific Data 数据集论文 *SEM, EDS and optical images of coffee-ring salt residues from 125 synthetic tap-water compositions*（作者 Xiaoyan Li、Cuicui Jiang、Rebecca Lahr）的项目目录。数据集共 125 个化学配方，每个配方重复 5 次，合计 625 个样本；配方由 NaHCO₃（5 个水平，对应 Table_1 到 Table_5）与 CaCl₂、MgSO₄ 的 5×5 组合构成。每个样本包含一张 SEM 图（sem.jpg 及仪器参数 sem.txt）、一张手机光学照片（image.jpg）和七张 EDS 元素图（C、Ca、Cl、Mg、Na、O、S）。

本目录于 2026-10-10 从建模项目（原名 CNN-Vision-Transformer_Model_for_Elements_Concentration_Estimation_by_Coffee-Ring_Effect_Residue_Patterns，现为 `coffee-ring-water-chemistry`）的 `dataset_paper/` 整体移出，内部结构保持原样，因此分析脚本里的相对路径不需要改。建模项目自己的 `data/` 保留了一份同样的原始数据（逐文件大小比对一致），两个项目各自独立，删改任何一边都不会影响另一边。

## 目录说明

`data/` 是旧的“增强”工作副本（其中 8 个 sem.txt 被覆盖过），不入库，也不用于发布。对外发布的数据包由 `analysis/build_release.py` 从 `experiment/data/` 生成：`release/` 是解压后的目录结构（原始文件为硬链接，另有 metadata、derived、splits.csv 和 MD5SUMS），`release_zenodo/` 是上传用的 25 个 raw_Table_t_Samples_a-b.zip、tables.zip、README.md 和校验文件（目录名沿用旧称，实际发布在 figshare）；数据包说明写在 `analysis/release_README.md`。两个目录都不入 git，随时可以重建。

`experiment/` 存原始数据：`experiment/data/` 是每个液滴的原始 SEM、EDS、光学图像和 sem.txt（另含 2021-08-31 生成的派生图 *_processed.jpeg），`recipies/` 是各元素浓度表与配方原始 xlsx。旧的统计、机器学习和图像校正脚本已于 2026-10-14 删除（在 git 历史里），`experiment/data_corrected/` 是旧校正脚本的输出，未入库。所有现行分析在 `analysis/`。

`manuscript/` 是投 Scientific Data 的 Data Descriptor 稿件：manuscript.tex（Springer Nature sn-jnl 模板）、references.bib、figures/ 和已编译的 manuscript.pdf。八张图由 `analysis/figures/` 生成：图 1 用 `fig1.py`，图 2–8 用 `figs.py`，共用样式在 `style.py`（与姊妹数据集论文一致：Arial 7 pt、180 mm 通栏、按毫米定位版面）。

## 状态

论文尚未投稿，代码仓库为私有的 https://github.com/xiaoyanLi629/coffee-ring-crystallization-dataset 。数据集已上传 figshare（条目 34516434，DOI 10.6084/m9.figshare.34516434，2026-10-15 已公开发布 v1；figshare 屏蔽 AutoDL 服务器，上传在东京 VPS 上用 analysis/figshare_upload.py 完成）。还缺代码发布版的 DOI。图像数据不入 git（见 .gitignore），只跟踪代码、表格、结果和稿件。
