# ABIDE 多站点脑功能连接图分类

本项目从 ABIDE I 的预处理 CC200 脑区时间序列构建静息态功能连接图，比较三类 ASD/典型发育对照组分类器，并用留一站点验证测试跨站点泛化。下载与特征准备、训练评估、模型预测分别通过独立脚本执行。CUDA 版 PyTorch 可用于训练轻量 GCN。

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

    abide_connect/
      PLAN.md
      README.md
      requirements.txt
      download_data.py
      train_model.py
      use_model.py
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
        models/
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

激活已配置好的 Conda 环境并进入项目根目录。项目依赖列在 requirements.txt 中，CUDA 版 PyTorch 需按当前系统和驱动选择匹配版本。

    conda activate abide
    python -m pip install -r requirements.txt

验证 PyTorch 与 CUDA。

    python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU mode')"

## 分阶段运行

下载、训练和预测是三个独立入口。首次下载需要联网，不会下载原始 MRI。数据准备脚本会保存 Nilearn 下载文件和派生特征缓存。中断后重新运行会复用已下载文件。

### 1. 下载和准备数据

    python download_data.py --data-dir data --top-k 20

这一步获取经过质量筛选的 ABIDE PCP CC200 ROI 时序，计算功能连接图、节点特征和 Fisher z 连接特征，并写入 data/processed。下载时终端显示已完成的 ROI 时序文件数、累计传输量和实时网速。已缓存文件会快速计入完成进度，不会重复下载。

若只需重建派生特征缓存，可运行：

    python download_data.py --data-dir data --top-k 20 --rebuild-cache

更改 top-k 时需使用 --rebuild-cache。Nilearn 已下载的原始数据会继续复用。

### 2. 训练和评估

训练脚本只读取已经准备好的缓存，不会触发下载。它运行留一站点验证及三类模型比较，然后根据各站点 GCN 的最佳轮数中位数，在完整数据上拟合并保存最终 GCN。

    python train_model.py --data-dir data --output-dir outputs --device auto

常用参数包括 --device auto/cuda/cpu、--batch-size 32、--max-epochs 100、--patience 12、--hidden-dim 32、--top-k 20 和 --seed 42。若找不到特征缓存，先运行 download_data.py。top-k 必须与生成缓存时一致。

训练结束后，最终模型保存在 outputs/models/signed_gcn.pt，交叉验证表格和图表保存在 outputs/results 和 outputs/figures。

### 3. 使用已训练模型进行预测

预测脚本读取 checkpoint 和一个派生特征 NPZ 缓存，输出 ASD 概率与 0.5 阈值下的预测标签。

    python use_model.py --model outputs/models/signed_gcn.pt --input data/processed/abide_cc200_graphs.npz --output outputs/results/predictions.csv --device auto

输入缓存需包含 node_features、positive_adjacency、negative_adjacency 和 covariates，形状分别为 N×200×5、N×200×200、N×200×200 和 N×3。若包含 subject_ids，预测表会保留受试者 ID。输入缓存和模型的 top-k 设置必须相同。使用同一 ABIDE 缓存训练并预测得到的是训练集预测，跨站点泛化性能应查看训练阶段生成的折外预测结果。

各脚本可用 --help 查看参数，不会在显示帮助时下载数据或训练模型。

## 数据与质控

Nilearn 参数固定为 CPAC、CC200、带通滤波开启、全局信号回归关闭、quality_checked 开启。根据 Nilearn 的质控规则筛选表型记录，并额外要求标签、年龄、性别、平均帧位移和 ROI 时间序列可用。缓存中的 participants.csv 是最终样本清单，应作为报告样本数的依据。

ABIDE 标记中 DX_GROUP 为 1 表示 ASD，为 2 表示典型发育对照。项目转换为 1 和 0。SEX 为二元编码，当前实现把原始值 2 设为 1，把原始值 1 设为 0。

## 输出文件

    outputs/
      models/
        signed_gcn.pt
      results/
        metrics_by_site.csv
        summary_metrics.csv
        oof_predictions.csv
        gcn_training_by_site.csv
        predictions.csv
      figures/
        mean_fc_by_group.png
        roc_oof.png
        site_auc.png

    data/processed/
      abide_cc200_graphs.npz
      participants.csv
      site_group_counts.csv

summary_metrics.csv 是总体折外指标。metrics_by_site.csv 按模型和留出站点列出表现。oof_predictions.csv 保存每位受试者在其未参与训练的站点折中的预测概率。gcn_training_by_site.csv 记录各折最佳验证轮数与损失。participants.csv 和 site_group_counts.csv 用于报告最终样本筛选和站点分布。

.gitignore 会忽略 data/raw、data/processed 和 outputs 下生成的下载文件、特征缓存、模型权重、结果表格与图像，不会将这些数据产物提交到 Git。

## GPU 使用

device 为 auto 且 CUDA 可用时，GCN 模型和每个批次的数据会被移动到 CUDA 设备。终端会打印设备名称。传统逻辑回归由 scikit-learn 在 CPU 上运行。功能连接计算也在 CPU 上完成，因为该数据规模下搬运到 GPU 不一定更快。

RTX 5070 Ti 等具备 CUDA 支持的 NVIDIA GPU 可以训练本项目的 GCN，但约 871 个受试者仍属于小规模图数据。建议先运行一个完整交叉验证，再按需要增加随机种子或敏感性分析。

建议预留 16 GB 系统内存和 5 GB 磁盘空间。预处理 ROI 时序约 0.3–0.5 GB。首次下载预计 5–30 分钟，连接矩阵计算约 2–10 分钟，完整的 20 站点 GCN 评估约预留 10–60 分钟。后两项训练耗时是根据数据规模估计，尚未在 RTX 5090 上实测。环境配置与数据检查约需 3–7 小时人工时间，报告撰写另需约 3–6 小时。

连接热图的坐标是 CC200 ROI 编号。本项目不包含独立的解剖标签映射文件，因此热图不直接标出脑区名称。

## 结果解读

若 GCN 的折外 AUROC 或平衡准确率高于传统基线，可描述为在本数据和站点留出方案下表现更好。若没有提高，也应如实报告。由于样本来源、扫描仪和站点分布不同，站点级结果可能变化较大。

## 本次评估结果

当前记录的完整运行纳入 871 名受试者和 20 个站点，其中 ASD 403 人、典型发育对照 468 人。留一站点折外结果如下。

| 模型 | AUROC | 平衡准确率 | 敏感度 | 特异度 |
|---|---:|---:|---:|---:|
| 协变量逻辑回归 | 0.578 | 0.548 | 0.434 | 0.662 |
| 功能连接 + 协变量逻辑回归 | 0.736 | 0.682 | 0.643 | 0.722 |
| 有符号 GCN | 0.533 | 0.523 | 0.566 | 0.481 |

在这次数据和留一站点验证设置下，功能连接与协变量组合的逻辑回归取得最高折外 AUROC 和平衡准确率。当前 GCN 未显示出高于逻辑回归基线的跨站点分类表现。该结论描述的是公开数据上的分组预测结果，不用于个体诊断或因果推断。站点样本量为 11–172 人，解释总体表现时应同时查看站点级指标。

可复核的结果表、折外预测、站点分布和图表保存在 [results/](results/)。`oof_predictions.csv` 是跨站点评估使用的预测文件。训练结束后生成的 `outputs/results/predictions.csv` 是独立预测入口的结果，不含真实标签。

## 主要参考

- Di Martino et al. The autism brain imaging data exchange: towards a large-scale evaluation of the intrinsic brain architecture in autism. Molecular Psychiatry, 2014. DOI: 10.1038/mp.2013.78.
- Abraham et al. Deriving reproducible biomarkers from multi-site resting-state data: An Autism-based example. NeuroImage, 2017. DOI: 10.1016/j.neuroimage.2016.10.045.
- Power et al. Spurious but systematic correlations in functional connectivity MRI networks arise from subject motion. NeuroImage, 2012. DOI: 10.1016/j.neuroimage.2011.10.018.
- ABIDE PCP Nilearn 数据接口：https://nilearn.github.io/stable/modules/generated/nilearn.datasets.fetch_abide_pcp.html

## 学术规范

实验设计类报告不少于 3000 字，需包括背景与研究进展、实验设计/材料与方法、变量与统计策略、关键结果和图表、讨论、科学意义、局限性、展望、课程关联和个人贡献。正文规范标注引用，文末列出参考文献，按学术出版物格式排版并导出 PDF，文件名使用学号和姓名。30 篇以上参考文献适用于综述类报告，本项目采用实验设计类报告。下载公开数据仍需正确引用数据集与预处理流程。不要将项目输出解释为临床诊断。
