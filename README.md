# ABIDE 多站点脑功能连接图分类

本项目从 ABIDE I 的预处理 CC200 脑区时间序列构建静息态功能连接图，比较三类 ASD/典型发育对照组分类器，并用留一站点验证测试跨站点泛化。项目按数据准备、模型、实验评估分模块实现，可用 CUDA 版 PyTorch 在 RTX 5090 上训练轻量 GCN。

## 研究问题

在年龄、性别和平均帧位移之外，功能连接是否提供 ASD 分组预测信息？在未参与训练的扫描站点上，有符号加权 GCN 是否比传统逻辑回归更好地利用脑网络结构？

这是一项公开数据上的分组预测研究。模型性能不用于个体诊断，也不代表功能连接造成疾病。

## 方法概述

每位受试者表示为包含 200 个脑区节点的图。脑区时间序列两两计算 Pearson 相关作为有符号边权。GCN 输入每个脑区的时序摘要特征、分开的正负连接矩阵和年龄/性别/平均帧位移。默认保留每个节点绝对相关最高的 20 条连接。

比较以下模型：

1. 年龄、性别和平均帧位移的逻辑回归。
2. Fisher z 功能连接特征与协变量的 L2 逻辑回归。
3. 正负边分通道的两层 GCN，并融合协变量。

外层使用留一站点验证。每轮一个站点只用于测试，GCN 的早停验证也按站点分组。早停选定训练轮数后，模型会在整个外层训练集上重新训练，再预测留出站点。结果包括 AUROC、平衡准确率、敏感度、特异度、站点级指标、折外预测和功能连接热图。

## 项目结构

    abide_connectome_gcn/
      PLAN.md
      README.md
      requirements.txt
      run_project.py
      data/
        raw/
        processed/
      src/
        abide_gcn/
          __init__.py
          data.py
          model.py
          experiment.py
      outputs/
        results/
        figures/

PLAN.md 包含研究设计、步骤、验收条件、结果清单和报告要求。

## 环境要求

- Windows 10/11、Linux 或 macOS。
- Python 3.11 推荐。
- 16 GB 系统内存可完成分析，32 GB 更舒适。
- 约 5 GB 可用磁盘空间，用于环境、下载缓存和结果。
- NVIDIA GPU 可选。RTX 5090 的显存远高于本项目需要，GCN 能在 CUDA 上训练。

项目不依赖 torch-geometric，GCN 使用 PyTorch 批量矩阵运算实现。

## 安装

在项目根目录执行以下命令。

    py -3.11 -m venv .venv
    .\.venv\Scripts\Activate.ps1
    python -m pip install --upgrade pip

先从 PyTorch 官方安装页选择匹配当前驱动的 Windows CUDA 版 PyTorch。RTX 5090 需要支持 Blackwell 架构的近期 PyTorch/CUDA 组合。示例安装命令如下，若官方推荐版本更新，以官方选择器为准。

    python -m pip install torch --index-url https://download.pytorch.org/whl/cu128
    python -m pip install -r requirements.txt

验证 CUDA 识别情况。

    python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU mode')"

如果 torch.cuda.is_available() 为 False，脚本可用 CPU 运行。检查 PyTorch 版本、显卡驱动和 CUDA wheel 是否匹配。

## 运行

第一次运行会通过 Nilearn 下载预处理后的 ROI 时间序列和表型数据。无需下载原始 MRI。首次下载需要联网，下载后会在 data/processed 生成可复用缓存。

    python run_project.py --data-dir data --output-dir outputs --device auto

常用参数：

    --data-dir       Nilearn 下载缓存根目录，默认 data
    --output-dir     结果目录，默认 outputs
    --device         auto、cuda 或 cpu，默认 auto
    --batch-size     GCN 批量大小，默认 32
    --max-epochs     每个外层折的最大训练轮数，默认 100
    --patience       早停轮数，默认 12
    --top-k          每个脑区保留的连接数，默认 20
    --seed           随机种子，默认 42
    --rebuild-cache  强制重建派生缓存，不会清除 Nilearn 下载文件

更改 --top-k 后需同时传入 --rebuild-cache，以便重新构造图邻接矩阵。Nilearn 已下载的原始 ROI 时序会继续复用。

CPU 后备运行示例：

    python run_project.py --data-dir data --output-dir outputs --device cpu

如需仅检查命令行参数，不会触发数据下载：

    python run_project.py --help

## 数据与质控

Nilearn 参数固定为 CPAC、CC200、带通滤波开启、全局信号回归关闭、quality_checked 开启。根据 Nilearn 的质控规则筛选表型记录，并额外要求标签、年龄、性别、平均帧位移和 ROI 时间序列可用。缓存中的 participants.csv 是最终样本清单，应作为报告样本数的依据。

ABIDE 标记中 DX_GROUP 为 1 表示 ASD，为 2 表示典型发育对照。项目转换为 1 和 0。SEX 为二元编码，当前实现把原始值 2 设为 1，把原始值 1 设为 0。

## 输出文件

    outputs/
      results/
        metrics_by_site.csv
        summary_metrics.csv
        oof_predictions.csv
        gcn_training_by_site.csv
      figures/
        mean_fc_by_group.png
        roc_oof.png
        site_auc.png

    data/processed/
      abide_cc200_graphs.npz
      participants.csv
      site_group_counts.csv

summary_metrics.csv 是总体折外指标。metrics_by_site.csv 按模型和留出站点列出表现。oof_predictions.csv 保存每位受试者在其未参与训练的站点折中的预测概率。gcn_training_by_site.csv 记录各折最佳验证轮数与损失。participants.csv 和 site_group_counts.csv 用于报告最终样本筛选和站点分布。

## GPU 使用

device 为 auto 且 CUDA 可用时，GCN 模型和每个批次的数据会被移动到 CUDA 设备。终端会打印设备名称。传统逻辑回归由 scikit-learn 在 CPU 上运行。功能连接计算也在 CPU 上完成，因为该数据规模下搬运到 GPU 不一定更快。

RTX 5090 可以训练本项目的 GCN，但约 871 个受试者仍属于小规模图数据。建议先运行一个完整交叉验证，再按需要增加随机种子或敏感性分析。

建议预留 16 GB 系统内存和 5 GB 磁盘空间。预处理 ROI 时序约 0.3–0.5 GB。首次下载预计 5–30 分钟，连接矩阵计算约 2–10 分钟，完整的 20 站点 GCN 评估约预留 10–60 分钟。后两项训练耗时是根据数据规模估计，尚未在 RTX 5090 上实测。环境配置与数据检查约需 3–7 小时人工时间，报告撰写另需约 3–6 小时。

连接热图的坐标是 CC200 ROI 编号。本项目不包含独立的解剖标签映射文件，因此热图不直接标出脑区名称。

## 结果解读

若 GCN 的折外 AUROC 或平衡准确率高于传统基线，可描述为在本数据和站点留出方案下表现更好。若没有提高，也应如实报告。由于样本来源、扫描仪和站点分布不同，站点级结果可能变化较大。

## 主要参考

- Di Martino et al. The autism brain imaging data exchange: towards a large-scale evaluation of the intrinsic brain architecture in autism. Molecular Psychiatry, 2014. DOI: 10.1038/mp.2013.78.
- Abraham et al. Deriving reproducible biomarkers from multi-site resting-state data: An Autism-based example. NeuroImage, 2017. DOI: 10.1016/j.neuroimage.2016.10.045.
- Power et al. Spurious but systematic correlations in functional connectivity MRI networks arise from subject motion. NeuroImage, 2012. DOI: 10.1016/j.neuroimage.2011.10.018.
- ABIDE PCP Nilearn 数据接口：https://nilearn.github.io/stable/modules/generated/nilearn.datasets.fetch_abide_pcp.html

## 学术规范

实验设计类报告不少于 3000 字，需包括背景与研究进展、实验设计/材料与方法、变量与统计策略、关键结果和图表、讨论、科学意义、局限性、展望、课程关联和个人贡献。正文规范标注引用，文末列出参考文献，按学术出版物格式排版并导出 PDF，文件名使用学号和姓名。30 篇以上参考文献适用于综述类报告，本项目采用实验设计类报告。下载公开数据仍需正确引用数据集与预处理流程。不要将项目输出解释为临床诊断。
