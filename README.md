# OpenVLA-Drive 🚗🤖

**基于视觉-语言-动作（VLA）模型的开源自动驾驶项目**

## 项目简介

OpenVLA-Drive 是一个创新的自动驾驶研究项目，利用 Vision-Language-Action (VLA) 模型在 CARLA 仿真器中实现端到端的自动驾驶控制。该项目旨在探索多模态基础模型在自动驾驶领域的应用潜力。

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

# 安装依赖
pip install -r requirements.txt
```

### 2. 安装 CARLA

下载并安装 CARLA 0.9.15:
```bash
# 下载地址: https://github.com/carla-simulator/carla/releases/tag/0.9.15
# 解压后设置环境变量
export CARLA_ROOT=/path/to/CARLA_0.9.15
```

### 3. 数据收集

```bash
# 启动 CARLA 服务器
cd $CARLA_ROOT
./CarlaUE4.sh

# 在另一个终端收集数据
python scripts/collect_data.py
```

### 4. 模型训练

```bash
python scripts/train.py --config configs/training/default.yaml
```

### 5. 闭环评估

```bash
python evaluation/closed_loop_sim.py --checkpoint path/to/checkpoint.ckpt
```

## 核心特性

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
- [x] 实现 VLA Driving Policy 模型架构
  - [x] 集成预训练 VLM backbone（LLaVA/Phi-3-Vision）
  - [x] CLIP 视觉编码器
  - [x] LoRA 适配器配置
  - [x] MLP 动作头（轨迹预测）
- [x] CARLA VLA 数据集加载器
  - [x] 图像预处理（CLIP 归一化）
  - [x] 文本 tokenization
  - [x] 轨迹归一化和重采样
  - [x] Custom collate function
- [x] PyTorch Lightning 训练模块
- [x] 评估指标和闭环仿真框架
- [x] 配置文件系统
- [x] 示例和测试脚本

### 进行中 🚧
- [ ] CARLA 数据收集脚本实现
- [ ] 在 CARLA 上收集驾驶数据
- [ ] 模仿学习训练
- [ ] 完整的闭环评估

### 计划中 📋
- [ ] 多任务学习（导航、避障、车道保持）
- [ ] 强化学习微调
- [ ] 预训练模型发布
- [ ] 性能优化和加速

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

欢迎提交 Issue 和 Pull Request！

## 许可证

MIT License

## 联系方式

- **作者**: 欧林海
- **邮箱**: franka907@126.com

---

**免责声明**: 本项目仅用于学术研究和教育目的，不应直接用于真实车辆的自动驾驶系统。
