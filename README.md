# OpenVLA-Drive

**基于视觉-语言-动作（VLA）模型的自动驾驶研究项目**

> **说明**: 本项目为个人研究项目，代码仅供学习参考。

## 项目简介

OpenVLA-Drive 探索 Vision-Language-Action (VLA) 模型在 CARLA 仿真器中的端到端驾驶应用。

## VLA 模型在自动驾驶中的概念

### 什么是 VLA 模型？

Vision-Language-Action (VLA) 模型是一种多模态深度学习架构，它能够：

1. **Vision (视觉)**: 处理来自摄像头的图像数据，理解场景语义
2. **Language (语言)**: 接收自然语言指令或生成驾驶相关的描述
3. **Action (动作)**: 输出车辆控制指令（转向、油门、刹车）

### VLA 在自动驾驶中的优势

- **端到端学习**: 直接从原始传感器数据到控制指令，无需手工设计中间表示
- **语言理解能力**: 可以理解自然语言导航指令（如"在下一个路口左转"）
- **泛化能力**: 预训练的视觉-语言模型带来更强的场景理解和泛化能力
- **可解释性**: 可以生成驾驶决策的语言描述，提高系统透明度

### 工作流程

```
摄像头图像 + 导航指令 → VLA 模型 → 车辆控制动作
                      ↓
                  场景理解 & 决策解释
```

## 技术栈

- **Python**: 3.10
- **深度学习框架**: PyTorch 2.1+
- **训练框架**: PyTorch Lightning 2.1+
- **模型库**: HuggingFace Transformers 4.35+
- **仿真器**: CARLA 0.9.15
- **配置管理**: Hydra + OmegaConf

## 项目结构

```
OpenVLA-Drive/
├── configs/              # 配置文件 (YAML)
│   ├── model_config.yaml      # VLA 模型配置
│   ├── data_config.yaml       # 数据集配置
│   ├── training_config.yaml   # 训练超参数配置
│   └── policy_config.yaml     # VLA Driving Policy 配置
├── data/                # 数据加载与预处理
│   ├── carla_dataset.py # CARLA VLA 数据集加载器 ✓
│   └── DATA_FORMAT.txt  # 数据格式规范
├── models/              # VLA 模型架构
│   ├── vla_model.py     # 基础 VLA 模型 ✓
│   └── policy.py        # VLA Driving Policy (LoRA + 轨迹预测) ✓
├── training/            # 训练相关代码
│   ├── lightning_module.py        # 基础 Lightning 模块 ✓
│   └── policy_lightning_module.py # Policy Lightning 模块 ✓
├── evaluation/          # 评估脚本
│   ├── closed_loop_sim.py # 闭环仿真评估框架 ✓
│   └── metrics.py       # 评估指标 ✓
├── scripts/             # 实用脚本
│   └── train.py         # 训练入口脚本 ✓
├── examples/            # 示例代码
│   ├── test_policy.py   # VLA Policy 测试示例 ✓
│   └── test_dataset.py  # 数据集测试示例 ✓
├── utils/               # 工具函数
├── requirements.txt     # Python 依赖 ✓
├── check_setup.py       # 环境检查脚本 ✓
└── README.md           # 项目文档
```

## 快速开始

### 1. 环境安装

```bash
# 创建虚拟环境
conda create -n openvla-drive python=3.10
conda activate openvla-drive

# 克隆项目
git clone https://github.com/olh2012/OpenVLA-Drive.git
cd OpenVLA-Drive

# 安装依赖
pip install -r requirements.txt

# 验证安装
python check_setup.py
```

### 2. 安装 CARLA（可选）

如需收集数据或进行仿真评估，需要安装 CARLA：

```bash
# 下载 CARLA 0.9.15
# 地址: https://github.com/carla-simulator/carla/releases/tag/0.9.15

# 解压后设置环境变量
export CARLA_ROOT=/path/to/CARLA_0.9.15
export PYTHONPATH=$PYTHONPATH:$CARLA_ROOT/PythonAPI/carla/dist/carla-0.9.15-py3.10-linux-x86_64.egg
```

### 3. 快速测试

**测试 VLA 驾驶策略模型**:

```bash
# 运行策略模型测试示例
python examples/test_policy.py

# 预期输出：
# - 模型架构总结
# - 可训练参数统计（仅 LoRA + Action Head）
# - 推理示例
# - 训练示例
```

**测试数据集加载器**:

```bash
# 运行数据集测试示例（会自动创建 dummy 数据）
python examples/test_dataset.py

# 预期输出：
# - 创建测试数据集
# - 加载和预处理示例
# - DataLoader 批处理示例
# - 数据格式验证
```

## 详细使用说明

### 模型使用

#### VLA Driving Policy 基本用法

```python
from models.policy import VLADrivingPolicy
import torch

# 1. 初始化模型
model = VLADrivingPolicy(
    model_name="microsoft/phi-2",  # 或 "llava-hf/llava-1.5-7b-hf"
    vision_model_name="openai/clip-vit-base-patch32",
    num_timesteps=10,  # 预测 10 个未来路径点
    use_lora=True,     # 使用 LoRA 高效微调
    lora_config={
        'r': 16,
        'lora_alpha': 32,
        'lora_dropout': 0.05,
    },
)

# 2. 准备输入
images = torch.randn(2, 3, 224, 224)  # [batch_size, C, H, W]
instructions = [
    "Follow the lane and maintain safe distance",
    "Turn left at the next intersection"
]

# 3. 推理
trajectory = model.predict_trajectory(
    image_tensors=images,
    text_instructions=instructions
)
print(f"Predicted trajectory: {trajectory.shape}")  # [2, 10, 2]

# 4. 查看可训练参数
model.print_trainable_parameters()
# 输出示例: Trainable: 2.5M / All: 2.7B (0.09%)
```

#### 多任务辅助输出

`VLADrivingPolicy` 支持多任务头（导航分类、避障得分、车道偏移等）。可在 `configs/policy_config.yaml -> model.multi_task` 中开启，推理时会额外返回：

```python
outputs = model.predict_trajectory(
    image_tensors=images,
    text_instructions=instructions,
    return_aux=True,
)
trajectory = outputs['trajectory']          # [B, T, 2]
multi_task = outputs.get('multi_task', {})  # dict: navigation_logits / lane_offset / obstacle_score
labels = outputs.get('navigation_labels', [])
```

示例 `examples/test_policy.py` 已展示如何读取这些信号，便于后续多任务损失或可解释性分析。

### 数据准备

#### CARLA 数据格式

请参考 `data/DATA_FORMAT.txt` 了解详细的数据格式规范。

**目录结构**:
```
datasets/carla/
├── train/
│   ├── images/
│   │   ├── 000000.png
│   │   └── ...
│   └── annotations.json
├── val/
│   └── ...
└── test/
    └── ...
```

**annotations.json 格式**:
```json
{
  "000000": {
    "image": "images/000000.png",
    "command": "Follow the lane",
    "trajectory": [[0.0, 0.0], [2.0, 0.1], [4.0, 0.3], ...],
    "ego_position": [x, y, theta]
  }
}
```

#### 使用数据加载器

```python
from data.carla_dataset import get_carla_vla_dataloader

# 创建 DataLoader
dataloader = get_carla_vla_dataloader(
    data_root='./datasets/carla',
    split='train',
    batch_size=8,
    tokenizer_name='microsoft/phi-2',
    num_trajectory_points=10,
    num_workers=4,
)

# 迭代数据
for batch in dataloader:
    images = batch['image']           # [8, 3, 224, 224]
    trajectories = batch['trajectory'] # [8, 10, 2]
    input_ids = batch['input_ids']    # [8, 128]
    attention_mask = batch['attention_mask']
    
    # 训练循环
    # ...
```

#### 数据自动采集脚本（进行中任务）

`scripts/collect_carla_data.py` 提供同步模式采集管线，可一键录制 CARLA 轨迹或在无仿真环境时生成伪数据：

```bash
# 在线采集（需要 CARLA Server）
python scripts/collect_carla_data.py \
  --host localhost --port 2000 \
  --town Town05 \
  --episodes 5 \
  --frames-per-episode 800 \
  --output-dir ./datasets/carla

# 离线演示（自动生成伪数据）
python scripts/collect_carla_data.py --offline --episodes 1 --frames-per-episode 50
```

脚本会按照 `data/DATA_FORMAT.txt` 生成 `images/` 与 `annotations.json`，可直接被 `CARLAVLADataset` 读取。

### 模型训练

#### 使用 PyTorch Lightning 训练

```python
import pytorch_lightning as pl
from training.policy_lightning_module import VLAPolicyLightningModule
from data.carla_dataset import get_carla_vla_dataloader
import yaml

# 1. 加载配置
with open('configs/policy_config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# 2. 准备数据
train_loader = get_carla_vla_dataloader(
    data_root='./datasets/carla',
    split='train',
    batch_size=config['training']['batch_size'],
    num_workers=4,
)

val_loader = get_carla_vla_dataloader(
    data_root='./datasets/carla',
    split='val',
    batch_size=config['training']['batch_size'],
    num_workers=4,
)

# 3. 创建模型
model = VLAPolicyLightningModule(
    model_config=config['model'],
    optimizer_config=config['training']['optimizer'],
    scheduler_config=config['training'].get('scheduler', {}),
    loss_config=config['training']['loss'],
)

# 4. 配置训练器
trainer = pl.Trainer(
    max_epochs=50,
    accelerator='gpu',
    devices=1,
    precision='16-mixed',
    gradient_clip_val=1.0,
)

# 5. 开始训练
trainer.fit(model, train_loader, val_loader)
```

**或使用命令行**:
```bash
python scripts/train.py --config configs/policy_config.yaml
```

### 评估和推理

```bash
# 在 CARLA 中进行闭环评估（自动统计碰撞/越线/路线完成度）
python evaluation/closed_loop_sim.py \
    --checkpoint checkpoints/best_model.ckpt \
    --host localhost \
    --port 2000 \
    --num-episodes 5 \
    --max-steps 800
```

> 若本地未安装 CARLA，脚本会退化为离线 mock 模式，方便快速验证模型推理逻辑。

#### 强化学习微调原型

为推进「强化学习微调」路线图，新增轻量环境 `training/rl_env.py` 以及示例脚本：

```bash
# Dry-run：仅检查依赖，不执行训练
python scripts/rl_finetune.py

# 需要安装 stable-baselines3+gymnasium 后可启动正式训练
python scripts/rl_finetune.py --vec-envs 4 --train-steps 10000
```

环境奖励由横向偏差/油门/刹车组合而成，可在无 CARLA 的情况下调试 RL 算法。

#### 模型导出（计划任务推进）

脚本 `scripts/export_policy.py` 可将训练好的 checkpoint 与配置打包，便于发布：

```bash
python scripts/export_policy.py \
  --checkpoint checkpoints/best_model.ckpt \
  --config configs/policy_config.yaml \
  --output-dir release/openvla_policy

# 若暂无权重，可使用 --mock 生成示例目录
python scripts/export_policy.py --mock
```

导出目录包含 `policy_state.pt`、`metadata.json` 及配置文件，可直接上传至 HuggingFace / ModelScope。

## 配置说明

### 模型配置 (configs/policy_config.yaml)

```yaml
model:
  backbone:
    model_name: "microsoft/phi-2"  # VLM backbone
    vision_model_name: "openai/clip-vit-base-patch32"
    freeze_vision_tower: true
    freeze_llm: true
  
  lora:
    use_lora: true
    r: 16              # LoRA rank
    lora_alpha: 32
    lora_dropout: 0.05
  
  action_head:
    num_timesteps: 10  # 预测的路径点数量
    hidden_dim: 512
    num_layers: 3

training:
  batch_size: 8
  learning_rate: 2.0e-4
  max_epochs: 50
```

## 配置参数

支持的预训练模型：Phi-2、LLaVA-1.5-7B、Phi-3-Vision

详细配置见 `configs/policy_config.yaml`

## 示例和教程

所有示例代码位于 `examples/` 目录：

- `test_policy.py`: VLA 驾驶策略模型完整测试
- `test_dataset.py`: 数据集加载和预处理示例

运行示例前请确保已安装所有依赖：
```bash
python check_setup.py
```

- ✅ 基于预训练 VLM 的 VLA 架构（支持 LLaVA、Phi-3-Vision）
- ✅ LoRA 高效微调（仅训练 <1% 参数）
- ✅ 轨迹预测（输出 T×2 未来路径点）
- ✅ CARLA 数据集加载器（支持图像+文本+轨迹）
- ✅ CLIP 图像预处理和归一化
- ✅ 自车坐标系轨迹归一化
- ✅ PyTorch Lightning 训练框架
- ✅ 支持多模态输入（视觉 + 语言指令）
- ✅ 闭环仿真评估框架
- ✅ 可配置的模型和训练参数

## 路线图

### 已完成 ✅
- ✅ 实现 VLA Driving Policy 模型架构
  - ✅ 集成预训练 VLM backbone（LLaVA/Phi-3-Vision）
  - ✅ CLIP 视觉编码器
  - ✅ LoRA 适配器配置
  - ✅ MLP 动作头（轨迹预测）
- ✅ CARLA VLA 数据集加载器
  - ✅ 图像预处理（CLIP 归一化）
  - ✅ 文本 tokenization
  - ✅ 轨迹归一化和重采样
  - ✅ Custom collate function
- ✅ PyTorch Lightning 训练模块
- ✅ 评估指标和闭环仿真框架
- ✅ 配置文件系统
- ✅ 示例和测试脚本

### 进行中 🚧
- ⏳ CARLA 数据收集脚本实现
  - `scripts/collect_carla_data.py` 已支持同步采集 + 离线伪数据 fallback
- ⏳ 在 CARLA 上收集驾驶数据
- ⏳ 模仿学习训练
- ⏳ 完整的闭环评估
  - `evaluation/closed_loop_sim.py` 接入 VLADrivingPolicy 并输出指标

### 计划中 📋
- 📌 多任务学习（导航、避障、车道保持）
  - ✅ 已提供可配置的多任务预测头，等待数据联合训练
- 📌 强化学习微调
  - ⚗️ `training/rl_env.py` + `scripts/rl_finetune.py` 可快速验证 PPO 管线
- 📌 预训练模型发布
  - 🧰 `scripts/export_policy.py` 可一键打包 state dict + metadata
- 📌 性能优化和加速

## 引用

如果您在研究中使用了本项目，请引用：

```bibtex
@misc{openvla-drive,
  author = {欧林海},
  title = {OpenVLA-Drive: Vision-Language-Action Models for Autonomous Driving},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/yourusername/OpenVLA-Drive}
}
```

## 贡献

本项目为个人研究项目，暂不接受外部贡献。

## 许可证

MIT License

## 作者

欧林海

---

**免责声明**: 本项目仅用于学术研究和教育目的，不应直接用于真实车辆的自动驾驶系统。
