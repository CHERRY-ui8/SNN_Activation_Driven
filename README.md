# SNN激活驱动反向传播算法实现（作业2阶段一）

本项目对应作业2中“阶段一：激活驱动的脉冲反向传播算法研究”任务，包含CIFAR10基础SNN训练、替代梯度函数对比、CIFAR10DVS神经形态数据集训练三大核心模块。


## 一、项目简介
- **核心目标**：实现激活驱动的脉冲神经网络（SNN）反向传播算法，对比替代梯度函数性能，适配神经形态数据集。
- **关键技术**：基于`spikingjelly`库构建卷积SNN，使用代理梯度（Surrogate Gradient）解决脉冲函数不可微问题。
- **支持任务**：
  1. CIFAR10数据集上的基础卷积SNN训练；
  2. 3种自定义替代梯度函数（Sigmoid、Esser、SuperSpike）性能对比；
  3. CIFAR10DVS数据集训练与预处理方式（按事件数/时间切分）对比。


## 二、环境依赖
### 1. 基础环境
- Python 3.8~3.10
- CUDA 11.6+（建议，GPU加速训练；无GPU可使用CPU）
- Conda（推荐使用conda管理环境）

### 2. 环境配置
**方式一：使用Conda环境文件（推荐）**
```bash
# 从environment.yml创建conda环境
conda env create -f environment.yml

# 激活环境
conda activate snn_oss
```

**方式二：手动安装（如不使用conda）**
```bash
# 安装PyTorch（含CUDA支持，根据自身CUDA版本调整）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 安装SNN核心库、数据处理与可视化库
pip install spikingjelly==0.0.0.0.14 pandas matplotlib tensorboard
```

**注意**：`environment.yml` 中已包含所有必需依赖（PyTorch、spikingjelly、pandas、matplotlib、tensorboard等）。如需GPU支持，可在激活环境后使用pip安装对应CUDA版本的PyTorch。


## 三、核心任务运行步骤
所有脚本均在项目根目录执行，可通过`--help`查看更多参数（如设备、批量大小、训练轮次）。

### 1. 任务1：CIFAR10基础SNN训练
**功能**：使用默认ATan替代梯度，实现CIFAR10数据集的卷积SNN训练，验证基础网络可行性。  
**运行命令**：
```bash
# 从项目根目录运行（确保已激活 conda 环境：conda activate snn_oss）
python src/train/train_cifar10_baseline.py \
  --device cuda:0 \
  --T 4 \
  --epochs 64 \
  --batch_size 128

# 参数说明：
# --device: 设备ID（如 cuda:0、cpu）
# --T: 时间步长（脉冲序列长度）
# --epochs: 训练轮次
# --batch_size: 批量大小（CPU建议改为32）
```
**输出结果**：
- 日志：`./logs/cifar10_baseline/`（含TensorBoard曲线）；
- 模型 checkpoint：`./logs/cifar10_baseline/checkpoints/`（保存最优模型`best.pth`）。


### 2. 任务2：替代梯度函数性能对比
**功能**：对比4种替代梯度（ATan默认、Sigmoid、Esser、SuperSpike）的准确率与收敛速度。  
**运行命令**：
```bash
python src/train/train_cifar10_surrogate_compare.py \
  --device cuda:0 \
  --T 4 \
  --epochs 64 \
  --batch_size 128 \
  --save_result ./results/surrogate_compare_result.csv
# --save_result: 对比结果保存路径（CSV表格）
```
**输出结果**：
- 对比表格：`./results/surrogate_compare_result.csv`（含最大准确率、收敛轮次、训练时间）；
- 各梯度日志：`./logs/cifar10_surrogate_compare/`（按梯度名称分目录）。


### 3. 任务3：CIFAR10DVS训练与预处理对比
**功能**：使用任务2选出的最优替代梯度（默认SuperSpike），对比CIFAR10DVS的两种预处理方式。  
**运行命令**：
```bash
python src/train/train_cifar10dvs.py \
  --device cuda:0 \
  --frame_num 16 \
  --epochs 32 \
  --batch_size 32 \
  --split_modes number,time \
    --save_result ./results/cifar10dvs_preprocess_compare.csv
# 参数说明：
# --frame_num: DVS切分帧数（时间步长T）
# --epochs: 训练轮次（DVS数据量大，轮次可减少）
# --batch_size: 批量大小（DVS数据占用显存高，建议≤32）
# --split_modes: 预处理方式（逗号分隔：number,time）
# --save_result: 预处理对比结果保存路径
```
**输出结果**：
- 对比表格：`./results/cifar10dvs_preprocess_compare.csv`；
- 日志与模型：`./logs/cifar10dvs_train/`（按预处理方式分目录）。


### 4. 附加任务：CIFAR10DVS高级预处理方法对比
**功能**：探索新的DVS数据预处理方法（事件计数归一化、时间表面变换、自适应归一化），并与基础方法对比。  
**运行命令**：
```bash
python src/train/train_cifar10dvs_advanced.py \
  --device cuda:0 \
  --frame_num 16 \
  --epochs 32 \
  --batch_size 32 \
  --methods baseline,count_norm,time_surface,adaptive_norm \
    --save_result ./results/cifar10dvs_advanced_preprocess_compare.csv
# 参数说明：
# --frame_num: DVS切分帧数（时间步长T）
# --epochs: 训练轮次
# --batch_size: 批量大小（建议≤32）
# --methods: 要对比的预处理方法（逗号分隔）
#   - baseline: 基础方法（按时间切分）
#   - count_norm: 事件计数归一化
#   - time_surface: 时间表面变换
#   - adaptive_norm: 自适应归一化
# --save_result: 预处理对比结果保存路径
```
**输出结果**：
- 对比表格：`./results/cifar10dvs_advanced_preprocess_compare.csv`；
- 各预处理方法日志：`./logs/cifar10dvs_advanced/`（按preprocess_method分目录）。


## 四、核心目录结构
```
SNN_Activation_Driven/
├─ src/                # 源代码
│  ├─ models/          # 网络模型（CIFAR10/CIFAR10DVS的SNN定义）
│  ├─ data/            # 数据加载（CIFAR10/CIFAR10DVS预处理）
│  ├─ surrogate/       # 自定义替代梯度函数
│  ├─ train/           # 训练脚本（对应三大任务）
│  ├─ utils/           # 通用工具（日志、模型保存等）
│  └─ analysis/        # 结果分析工具
├─ data/               # 数据集（自动下载到此处）
├─ logs/               # 训练日志与模型 checkpoint
├─ results/            # 实验结果表格（CSV文件）
├─ reports/            # 项目报告与图表
├─ environment.yml     # Conda环境配置文件
└─ surrogate.py        # spikingjelly替代梯度库文件
```


## 五、数据下载指南

本项目使用两个数据集：**CIFAR10**（静态图像）和**CIFAR10DVS**（神经形态事件流）。数据集会在首次运行时自动下载，也可手动下载。

### 1. CIFAR10数据集

- 首次运行训练脚本时，`torchvision`会自动从官方源下载CIFAR10数据集
- 下载位置：`./data/CIFAR10/`
- 数据集大小：约170MB
- 下载源：PyTorch官方镜像（通常较快）

### 2. CIFAR10DVS数据集

- 首次运行CIFAR10DVS训练脚本时，`spikingjelly`库会自动下载数据集
- 下载位置：`./data/CIFAR10DVS/`
- 数据集大小：约1.2GB（较大，建议确保网络稳定）
- 下载源：SpikingJelly官方维护的镜像

### 3. 数据集目录结构

下载完成后，`./data/`目录结构应如下：
```
data/
├─ CIFAR10/
│  └─ cifar-10-batches-py/
│     ├─ data_batch_1
│     ├─ data_batch_2
│     ├─ ...
│     └─ test_batch
└─ CIFAR10DVS/
   └─ [CIFAR10DVS数据文件]
```
## 六、注意事项
1. **环境激活**：运行训练脚本前，请确保已激活conda环境：
   ```bash
   conda activate snn_oss
   ```
2. **数据集下载**：CIFAR10/CIFAR10DVS会自动下载到`./data/`目录，DVS数据集约1.2GB，建议提前确保网络稳定。
3. **硬件适配**：
   - 无GPU时，将所有命令的`--device cuda:0`改为`--device cpu`，并减小`--batch_size`（如32）；
   - 显存不足时，可降低`--channels`（如从32改为16）或`--frame_num`（如从16改为8）。
4. **日志查看**：运行TensorBoard命令查看准确率/损失曲线（以任务1为例）：
   ```bash
   tensorboard --logdir ./logs/cifar10_baseline/
   ```
5. **作业报告素材**：任务2的`surrogate_gradients.png`（替代梯度曲线）、任务2/3的CSV表格（位于`results/`目录），可直接用于报告中的结果分析。
6. **结果分析**：使用`src/analysis/analyze_results.py`脚本分析训练结果，生成可视化图表。