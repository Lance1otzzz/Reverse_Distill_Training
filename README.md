# RDT 时间序列预测验证框架
update:2025/05/27
## 1. 项目概览

本项目提供一个基于 Python 和 PyTorch 的**时间序列预测验证框架**。其核心目标是为 **RDT (Reverse Distillation Training)** 方法在时间序列预测领域的应用提供一个**可复现、可扩展**的实验环境，以便系统性地验证其有效性。

### RDT 核心理念

传统知识蒸馏通常使用复杂模型（教师）指导简单模型（学生）。**RDT (Reverse Distillation Training)** 则提出一种**反向**思路：利用结构简单、训练稳定的时间序列模型（如 DLinear、ARIMA）作为**教师**，去指导结构复杂、容量更大的现代深度学习模型（如 PatchTST、Autoformer）作为**学生**。

这一方法的创新点在于：

*   **提升稳定性与鲁棒性**: 简单教师模型固有的时序归纳偏置和稳定性，可以作为正则化手段，约束复杂学生模型的过度自由度，缓解训练不稳定和过拟合问题。
*   **隐式注入先验知识**: 教师模型能更好地捕捉时间序列的基础模式（如趋势、周期性），通过蒸馏过程将这些知识传递给学生模型，弥补其可能缺乏的显式时序归纳偏置。
*   **动态平衡**: 通过调整任务损失与蒸馏损失的权重，可以在拟合真实数据（学生模型强项）和模仿教师基础模式（教师模型强项）之间取得动态平衡，优化模型的泛化能力。

### 技术方案

框架围绕 RDT 理念构建，包含以下关键组件和机制：

1.  **模型组件**: 支持集成多种时间序列模型作为教师（如 DLinear）和学生（如 PatchTST）。框架基于 `neuralforecast` 库预置了 DLinear 和 PatchTST 的支持。
2.  **复合训练目标**: 学生模型在训练过程中优化一个复合损失函数：

    $$L_{total} = \alpha \cdot L_{task}(Y_{student}, Y_{true}) + (1-\alpha) \cdot L_{distill}(Y_{student}, Y_{teacher})$$

    其中， $L_{task}$ 是学生预测 ( $Y_{student}$ ) 与真实值 ( $Y_{true}$ ) 之间的任务损失（如 MSE 或 MAE）， $L_{distill}$ 是学生预测 ( $Y_{student}$ ) 与教师预测 ( $Y_{teacher}$ ) 之间的蒸馏损失。
3.  **动态 Alpha 调度**: 参数 $\alpha$ 平衡了任务损失和蒸馏损失的重要性。框架支持 $\alpha$ 在训练过程中进行动态调整（Alpha Scheduling），例如初期 $\alpha$ 较小以更多依赖教师指导，后期 $\alpha$ 逐渐增大以更多拟合真实数据。
    > 注意：计算 $L_{distill}$ 时，教师模型的输出 $Y_{teacher}$ 会使用 `.detach()`，确保梯度只回传给学生模型。

该框架不仅实现了 RDT 训练流程，还提供了必要的基线模型（独立训练的教师、仅使用任务损失训练的学生 **Task-Only** ($\alpha=1$)、仅使用蒸馏损失训练的学生 **Follower** ($\alpha=0$)) 的训练和评估能力，便于进行全面的性能对比分析。此外，框架集成了数据处理、模型评估（包括鲁棒性测试）和结果可视化等配套功能。

## 2. 主要特性

*   **RDT 训练实现**: 精确实现基于复合损失和 Alpha 调度的 RDT 核心训练逻辑。
*   **全面的基线对比**: 支持训练并对比 RDT 模型与标准教师模型、Task-Only 学生模型 ($\alpha=1$)、Follower 学生模型 ($\alpha=0$) 的性能。
*   **灵活的模型集成**: 方便地引入和配置基于 PyTorch 的自定义或来自第三方库（如 `neuralforecast`）的其他时间序列模型。
*   **模块化与可配置性**: 清晰的代码结构 (数据、模型、训练器、评估器等) 和中心化的配置 (`src/config.py`)，易于理解、修改和扩展。
*   **标准数据处理流程**: 包含时间序列数据加载、预处理 (特征选择、缺失值、标准化)、划分 (训练/验证/测试) 和 `DataLoader` 构建。
*   **Alpha 调度策略**: 支持多种 Alpha 动态调整策略 (如 `linear`, `exponential`) 或固定值 (`constant`)。
*   **Alpha 门控机制**: 可选的 `GatedAlphaScheduler` 在验证损失长期无改善时自动终止学生模型训练。
*   **多维度评估**:
    *   计算常用预测指标 (MSE, MAE)。
    *   支持通过向测试数据添加噪声来评估模型的鲁棒性。
    *   支持多次运行实验以分析结果的稳定性。
    *   **学生-教师模型相似度评估**: 分析学生模型输出与教师模型输出之间的相似度。
*   **丰富的可视化**:
    *   自动绘制训练/验证损失曲线、Alpha 变化曲线。
    *   绘制真实值与各模型预测值的对比图。
    *   生成性能指标、鲁棒性测试结果的对比图表。
    *   绘制多次运行结果的稳定性分析图 (箱线图/小提琴图)。
*   **自动化结果管理**: 实验结果 (模型权重、评估指标 CSV、图表) 自动保存到结构化的 `results/` 目录。

## 3. 项目结构

```
rdt_framework/
│
├── main.py             # 项目入口，执行端到端实验流程
├── requirements.txt    # 项目依赖列表
├── README.md           # 本文档
│
├── data/               # 存放原始时间序列数据集
│   └── weatherHistory.csv  # 示例数据集 (请替换为您的数据)
│
├── results/            # 存放实验输出 (自动生成)
│   ├── metrics/        # 评估指标 CSV 文件
│   ├── models/         # 训练好的模型权重 (.pt)
│   └── plots/          # 生成的图表 (.png)
│
└── src/                # 源代码目录
    ├── __init__.py     # 使 src 可作为 Python 包导入
    ├── config.py       # 全局配置文件，所有参数设置
    ├── data_handler.py # 数据加载、预处理、划分与 DataLoader
    ├── models.py       # 模型定义与获取逻辑
    ├── trainer.py      # 训练器实现 (RDT, Standard, EarlyStopping)
    ├── schedulers.py   # Alpha 调度器实现
    ├── evaluator.py    # 模型评估、指标计算、鲁棒性测试
    └── utils.py        # 辅助函数 (种子设置、绘图、保存加载等)
```

## 4. 快速开始 (Getting Started)

遵循以下步骤设置环境、准备数据并运行您的第一个实验。

### 4.1 环境搭建

1.  **克隆项目仓库**:
    ```bash
    git clone <您的项目仓库地址>
    cd rdt_framework
    ```
2.  **创建并激活虚拟环境** (强烈推荐):
    ```bash
    python -m venv venv
    # Linux / macOS
    source venv/bin/activate
    # Windows
    # venv\Scripts\activate
    ```
3.  **安装依赖**: 确保您的环境中已安装 PyTorch (选择对应 CUDA 版本的版本)。然后安装项目依赖：
    ```bash
    pip install -r requirements.txt
    ```
    `requirements.txt` 文件内容示例如下 (版本号根据您的环境和需求调整):
    ```txt
    torch>=2.6.0+cu124 # 根据你的CUDA版本选择合适的torch版本
    pandas>=2.2.3
    numpy>=2.2.4
    scikit-learn>=1.6.1
    matplotlib>=3.10.1
    seaborn>=0.13.2
    neuralforecast>=3.0.0
    tqdm>=4.67.1
    statsmodels>=0.14.4
    ```

### 4.2 数据准备

将您的时间序列数据集（例如 CSV 文件）放置到项目根目录下的 `data/` 文件夹中。

### 4.3 配置实验

打开并编辑 `src/config.py` 文件。这是所有实验参数的集中管理地，您需要根据您的数据集、选择的模型和实验需求进行设置：

*   指定 `DATASET_PATH`, `DATE_COL`, `TARGET_COLS`, `TIME_FREQ` 等数据相关参数。
*   选择 `TEACHER_MODEL_NAME` 和 `STUDENT_MODEL_NAME`，并配置 `TEACHER_CONFIG`, `STUDENT_CONFIG` 字典中的模型超参数。
*   设置训练参数，如 `EPOCHS`, `LEARNING_RATE`, `BATCH_SIZE`, `DEVICE` 等。
*   配置 RDT 特有的参数，包括 `ALPHA_START`, `ALPHA_END`, `ALPHA_SCHEDULE`, `CONSTANT_ALPHA`。
*   设置评估参数，如 `METRICS` 列表和 `ROBUSTNESS_NOISE_LEVELS`。
*   配置实验管理参数，如 `SEED`, `EXPERIMENT_NAME`, `STABILITY_RUNS` (用于多次运行分析稳定性)。

请仔细检查 `config.py` 中的每一个参数，确保其与您的实验设定一致。

### 4.4 运行实验

在项目根目录下执行主脚本 `main.py`：

```bash
python main.py
```

脚本将按照 `config.py` 中的设置，依次执行数据加载、教师模型训练、基线学生模型训练、RDT 学生模型训练、模型评估（包括鲁棒性测试）、结果保存和图表生成等步骤。如果 `STABILITY_RUNS` 大于 1，整个过程将重复执行指定次数以进行稳定性分析。

## 5. 配置详情 (Configuration)

如前所述，所有实验参数都通过修改 `src/config.py` 文件来控制。主要配置项分组如下：

*   **路径设置**: `DATA_DIR`, `RESULTS_DIR` 等，定义数据和结果存放位置。
*   **数据集配置**: 数据文件路径，日期列、目标列名称，时间频率，回看窗口 (`LOOKBACK_WINDOW`)，预测范围 (`PREDICTION_HORIZON`)，以及训练/验证/测试集划分比例。
*   **数据处理配置**: 批次大小 (`BATCH_SIZE`) 等。
*   **模型配置**: 选择教师和学生模型的类型及其详细超参数。
*   **训练配置**: 训练设备 (`DEVICE`), 最大轮数 (`EPOCHS`), 优化器类型和参数 (`LEARNING_RATE`, `WEIGHT_DECAY`), Early Stopping 设置 (`PATIENCE`), 任务损失函数 (`LOSS_FN`)。
*   **RDT 配置**: Alpha 调度策略 (`ALPHA_SCHEDULE`), 起始/结束 Alpha 值 (`ALPHA_START`, `ALPHA_END`), 固定 Alpha 值 (`CONSTANT_ALPHA`), 以及可选的 Alpha 门控参数 (`USE_ALPHA_GATING`, `GATING_THRESHOLD`, `GATING_PATIENCE`)。
*   **评估配置**: 需要计算的指标列表 (`METRICS`), 鲁棒性测试的噪声水平列表 (`ROBUSTNESS_NOISE_LEVELS`)。
*   **实验管理**: 随机种子 (`SEED`) 保证可复现性, 实验名称 (`EXPERIMENT_NAME`), 稳定性运行次数 (`STABILITY_RUNS`)。

**在运行实验前，务必根据您的需求修改 `config.py` 文件。**

## 6. 核心概念解释

为了更好地理解项目，这里对几个关键概念进行阐述：

*   **教师模型 (Teacher Model)**: 在 RDT 中，通常指一个结构简单、训练稳定、具有良好时序归纳偏置的模型（如 DLinear）。它被首先独立训练，其预测结果用于指导学生模型。
*   **学生模型 (Student Model)**: 指容量较大、表达能力强的深度学习模型（如 PatchTST）。它是 RDT 训练的主要优化目标，学习同时拟合真实数据和模仿教师的预测。
*   **任务损失 (Task Loss, $L_{task}$)**: 学生模型预测与真实标签之间的差距，是衡量预测准确性的标准指标（如 MSE, MAE）。
*   **蒸馏损失 (Distillation Loss, $L_{distill}$)**: 学生模型预测与教师模型预测之间的差距。学生通过最小化此损失来学习教师模型的输出分布或行为。
*   **Alpha ($\alpha$)**: 复合损失函数中平衡任务损失和蒸馏损失的权重因子。$\alpha=1$ 对应仅使用任务损失的标准训练（Task-Only），$\alpha=0$ 对应仅使用蒸馏损失（Follower），而 RDT 则使用介于 0 和 1 之间的动态或固定 $\alpha$。
*   **Alpha 调度 (Alpha Scheduling)**: 在训练过程中动态调整 $\alpha$ 值的策略，例如线性增加或指数衰减，以适应不同的训练阶段需求。
*   **基线模型 (Baseline Models)**: 用于与 RDT 方法进行比较的模型：
    *   **独立训练的教师**: 评估简单教师模型的独立性能。
    *   **Task-Only 学生 ($\alpha=1$)**: 评估学生模型在没有蒸馏的标准训练下的性能。
    *   **Follower 学生 ($\alpha=0$)**: 评估学生模型仅通过模仿教师进行训练时的性能。

## 7. 实验输出

运行 `main.py` 脚本后，所有结果将组织并保存在 `results/` 目录下。主要输出文件类型包括：

*   **模型权重 (`results/models/`)**:
    *   `teacher_*.pt`: 训练好的教师模型权重。
    *   `student_*_task_only_*.pt`: Task-Only 学生模型权重。
    *   `student_*_rdt_*.pt`: RDT 学生模型权重。
    *   `student_*_follower_*.pt`: Follower 学生模型权重。
*   **图表 (`results/plots/`)**:
    *   训练损失曲线图（为不同训练模式生成）。
    *   Alpha 调度曲线图 (如果 Alpha 是动态的)。
    *   测试集预测对比图 (展示真实值与各模型的预测曲线)。
    *   性能指标对比图 (条形图对比各模型在测试集上的 MSE/MAE 等)。
    *   鲁棒性曲线图 (展示各模型在不同噪声水平下性能变化)。
    *   稳定性分析图 (箱线图/小提琴图展示多次运行的指标分布)。
*   **评估指标 (`results/metrics/`)**:
    *   实验总览 CSV 文件 (`{ExperimentName}_{Timestamp}.csv`)：包含每次运行的配置、模型路径和核心指标。
    *   鲁棒性详细结果 CSV 文件 (`{ModelName}_run*_robustness.csv`)：记录单一模型在不同噪声水平下的具体指标。
    *   稳定性分析总结 CSV 文件 (`{ExperimentName}_stability_summary_{Timestamp}.csv`)：(如果 `STABILITY_RUNS > 1`) 多次运行结果的均值和标准差汇总。

## 8. 扩展性

本框架设计时考虑了良好的可扩展性，您可以轻松地：

*   **添加新的时间序列模型**:
    1.  在 `src/models.py` 中实现或导入您的模型类。
    2.  修改 `get_model` 函数以识别新的模型名称并正确实例化。
    3.  在 `src/config.py` 中更新 `TEACHER_MODEL_NAME`/`STUDENT_MODEL_NAME` 和相应的配置字典。
*   **使用不同的数据集**:
    1.  将您的数据文件放入 `data/` 目录。
    2.  修改 `src/config.py` 中的数据相关参数 (`DATASET_PATH`, `DATE_COL`, `TARGET_COLS`, 等)。
    3.  如果数据格式或预处理逻辑与当前实现差异较大，可能需要调整 `src/data_handler.py`。
*   **实现新的 Alpha 调度策略**:
    1.  在 `src/schedulers.py` 中创建一个新的类，继承自 `BaseAlphaScheduler`。
    2.  修改 `get_alpha_scheduler` 函数以返回您的新调度器实例。
    3.  在 `src/config.py` 中设置 `ALPHA_SCHEDULE` 为您的新策略名称并配置相关参数。
*   **增加新的评估指标**:
    1.  在 `src/evaluator.py` 的 `calculate_metrics` 函数中添加新的指标计算逻辑。
    2.  在 `src/config.py` 的 `METRICS` 列表中加入新指标的名称。
    3.  (可选) 更新 `src/utils.py` 中的绘图函数以支持可视化新指标。

## 9. 贡献指南 (Contributing)

我们欢迎对本项目做出贡献！如果您有兴趣改进此框架，请遵循以下步骤：

1.  **Fork 项目**: 将本仓库 Fork 到您的 GitHub 账户。
2.  **创建分支**: 为您的新功能或 Bug 修复创建一个新的分支 (`git checkout -b feature/your-feature-name` 或 `bugfix/your-bug-fix-name`)。
3.  **提交更改**: 进行您的更改并提交 (`git commit -m "feat: Add new feature"` 或 `fix: Fix bug`)。请确保您的提交信息清晰明了。
4.  **推送分支**: 将您的分支推送到 Fork 的仓库 (`git push origin feature/your-feature-name`)。
5.  **创建 Pull Request**: 提交一个 Pull Request 到主仓库的 `main` 分支。请在 PR 描述中详细说明您的更改内容和目的。

在提交 Pull Request 之前，请确保您的代码通过了所有测试，并且遵循项目现有的编码风格。

## 10. 潜在的未来工作

*   **STL-RDT 融合**: 探索如何将时间序列的显式分解 (趋势、季节、残差) 与 RDT 相结合，例如对不同组件应用 RDT。
*   **多教师协同**: 研究使用多个不同类型的简单教师模型共同指导一个学生模型的策略。
*   **理论分析**: 对 RDT 在时间序列预测中的优化行为、泛化能力和隐式正则化效应进行更深入的理论探究。
*   **更多模型集成**: 扩展对更多流行时间序列模型的内置支持。


## 11. 许可证 (License)

```
MIT License

Copyright (c) 2025 Gibber1977

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## 12. 致谢 (Acknowledgments)

*   感谢 PyTorch 社区提供的深度学习框架。
*   感谢 `neuralforecast` 库提供了多种时间序列模型实现。
*   数据来源：[Autoformer - Google 云端硬盘](https://drive.google.com/drive/folders/1ZOYpTUa82_jCcxIdTmyr0LXQfvaM9vIy)
