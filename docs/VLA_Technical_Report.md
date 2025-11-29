# Vision-Language-Action (VLA) 模型在自动驾驶中的应用

**OpenVLA-Drive 技术报告**

---

**作者**: 欧林海  
**版本**: v1.0  
**日期**: 2025年1月

> **说明**: 本项目为个人研究项目，代码仅供学习参考。

---

## 摘要

本技术报告详细阐述了 OpenVLA-Drive 项目的技术架构、实现细节与评估方法。该项目探索了 Vision-Language-Action (VLA) 模型在端到端自动驾驶中的应用，通过集成预训练视觉-语言模型（如 LLaVA、Phi-3-Vision）与参数高效微调技术（LoRA），实现了基于自然语言指令的轨迹预测与车辆控制。

**关键词**: 视觉-语言-动作模型、端到端自动驾驶、LoRA微调、轨迹预测、CARLA仿真

---

## 1. 引言与研究背景

### 1.1 VLA 模型的机遇

Vision-Language-Action (VLA) 模型将视觉-语言模型扩展到具身智能领域：

```
[视觉输入] + [语言指令] → [VLA模型] → [动作输出]
```

在自动驾驶中，VLA 模型可以：
- 理解导航指令（如"在下一个路口左转"）
- 根据场景语义生成合理轨迹
- 提供语言解释增强透明度

### 1.2 OpenVLA-Drive 的定位

**项目目标**:
- 在 CARLA 仿真器中实现语言可控的端到端驾驶
- 输出未来轨迹路径点（T×2坐标）
- 最小化训练成本（<1% 参数更新，使用LoRA）

---

## 2. 问题定义

### 2.1 任务形式化

**输入**:
- 视觉观测: RGB图像 (H×W×3)
- 语言指令: 自然语言导航命令

**输出**:
- 未来轨迹: {(x₁, y₁), ..., (xₜ, yₜ)} (自车坐标系下的 T 个路径点)

---

## 3. 系统架构

### 3.1 整体流程

```
┌─────────────┐      ┌──────────────┐      ┌─────────────┐
│  RGB Image  │──┐   │              │      │  Trajectory │
│  (800×600)  │  │   │              │      │   (T×2)     │
└─────────────┘  ├──>│  VLA Policy  │─────>│  waypoints  │
                 │   │              │      └─────────────┘
┌─────────────┐  │   │              │
│   Command   │──┘   │              │
│  "左转..."  │      └──────────────┘
└─────────────┘
```

### 3.2 模块组成

**已实现模块** ✓:
- `configs/`: YAML配置文件（模型、数据、训练、策略）
- `data/carla_dataset.py`: CARLA数据集加载器
- `models/policy.py`: VLA Driving Policy核心实现
- `training/policy_lightning_module.py`: PyTorch Lightning训练模块
- `evaluation/`: 评估框架与指标
- `examples/`: 测试示例代码

---

## 4. 模型设计

### 4.1 VLA Driving Policy 架构

```python
class VLADrivingPolicy(nn.Module):
    components:
        1. vision_tower: CLIP视觉编码器
        2. llm_backbone: LLM语言编码器 (Phi-2/LLaVA)
        3. LoRA适配器: 参数高效微调
        4. vision_projection: 视觉-语言对齐
        5. action_head: MLP轨迹预测头
```

### 4.2 视觉编码器

- **模型**: CLIP ViT-Base/Large
- **输入**: 224×224 RGB图像（CLIP归一化）
- **输出**: 视觉特征向量
- **冻结策略**: 默认冻结，仅训练投影层

### 4.3 语言编码器

**支持的预训练模型**:
- Phi-2 (microsoft/phi-2): 2.7B参数，轻量级
- LLaVA-1.5-7B: 7B参数，更强理解能力  
- Phi-3-Vision: 支持长上下文

### 4.4 LoRA微调

**配置** (r=16):
```yaml
lora:
  r: 16              # 低秩维度
  lora_alpha: 32     # 缩放因子
  lora_dropout: 0.05
  target_modules: [q_proj, v_proj, k_proj, o_proj]
```

**参数统计**:
- Phi-2: 可训练 ~2.5M / 总计 2.7B (0.09%)
- LLaVA-7B: 可训练 ~6.7M / 总计 7B (0.096%)

### 4.5 动作头

3层MLP结构:
```python
Input [B, d_llm] 
  → Linear(d_llm, 512) → ReLU → Dropout
  → Linear(512, 512) → ReLU → Dropout  
  → Linear(512, T*2)
Output [B, T, 2]  # T个(x,y)路径点
```

---

## 5. 数据处理

### 5.1 CARLA数据格式

**目录结构**:
```
datasets/carla/
├── train/
│   ├── images/
│   │   ├── 000000.png
│   │   └── ...
│   └── annotations.json
├── val/
└── test/
```

**annotations.json**:
```json
{
  "000000": {
    "image": "images/000000.png",
    "command": "Follow the lane",
    "trajectory": [[0.0, 0.0], [2.0, 0.1], ...],
    "ego_position": [x, y, theta]
  }
}
```

### 5.2 数据预处理

**图像**: 
- Resize到224×224
- CLIP归一化: mean=[0.4815, 0.4578, 0.4082], std=[0.2686, 0.2613, 0.2758]

**文本**:
- Tokenization (最大长度128)
- Padding到固定长度

**轨迹**:
1. 归一化到自车坐标系（平移+旋转）
2. 重采样到固定T点（线性插值）

---

## 6. 训练方法

### 6.1 损失函数

**Smooth L1 Loss**:
```python
loss = F.smooth_l1_loss(pred_trajectory, gt_trajectory)
```

**评估指标**:
- ADE (Average Displacement Error)
- FDE (Final Displacement Error)

### 6.2 优化器

```yaml
optimizer:
  name: adamw
  lr: 2.0e-4
  weight_decay: 0.01
  
scheduler:
  name: cosine
  min_lr: 1.0e-6
```

### 6.3 训练技巧

- 混合精度训练 (FP16)
- 梯度裁剪 (clip_val=1.0)
- 仅训练LoRA适配器+动作头
- 冻结视觉塔和LLM主干

---

## 7. 评估体系

### 7.1 开环评估

在固定数据集上比较预测轨迹与ground truth:
- ADE/FDE指标
- 横向/纵向误差

### 7.2 闭环评估 (CARLA)

**指标**:
- 路线完成度
- 碰撞次数
- 闯红灯次数
- 离路百分比

**状态**: 框架已建立 🚧，CARLA控制接口为占位代码

---

## 8. 实现状态

### 8.1 已完成 ✓

- VLA Driving Policy模型架构
- CLIP视觉编码器集成
- LLM语言编码器（Phi-2/LLaVA支持）
- LoRA参数高效微调
- MLP轨迹预测头
- CARLA数据集加载器
- 图像CLIP归一化
- 文本tokenization
- 轨迹归一化到自车坐标系
- 轨迹重采样
- PyTorch Lightning训练框架
- 损失函数与优化器
- 评估指标（ADE/FDE）
- 闭环评估框架
- 示例代码与测试脚本

### 8.2 进行中 🚧

- CARLA闭环评估的完整实现
- CARLA数据收集脚本
- 实际驾驶数据集准备

### 8.3 未来工作 ⏳

- 语言生成解释
- 注意力可视化
- 安全约束层
- 鲁棒性验证
- 强化学习微调
- 真实车辆部署

---

## 9. 快速开始

### 9.1 环境安装

```bash
conda create -n openvla-drive python=3.10
conda activate openvla-drive
git clone https://github.com/olh2012/OpenVLA-Drive.git
cd OpenVLA-Drive
pip install -r requirements.txt
python check_setup.py
```

### 9.2 测试模型

```bash
# 测试VLA策略
python examples/test_policy.py

# 测试数据加载
python examples/test_dataset.py
```

### 9.3 训练（需准备数据）

```bash
python scripts/train.py --config configs/policy_config.yaml
```

---

## 10. 参考文献

### 视觉-语言模型
1. CLIP (Radford et al., 2021)
2. LLaVA (Liu et al., 2023)  
3. Phi-3-Vision (Microsoft, 2024)

### 端到端驾驶
4. PilotNet (Bojarski et al., 2016)
5. CIL (Codevilla et al., 2018)

### 参数高效微调
6. LoRA (Hu et al., 2021)
7. PEFT Library (HuggingFace)

### 机器人VLA
8. RT-1/RT-2 (Google DeepMind)
9. VIMA (Jiang et al., 2023)

### 仿真平台
10. CARLA (Dosovitskiy et al., 2017)

---

## 附录

### A. 配置文件示例

详见 `configs/policy_config.yaml`

### B. 常见问题

**Q: 显存不足？**  
A: 减小batch_size、使用混合精度、或选择Phi-2

**Q: 如何更换backbone？**  
A: 修改`configs/policy_config.yaml`中的`model_name`

**Q: 如何调整轨迹点数？**  
A: 修改`num_timesteps`参数

---

**项目地址**: https://github.com/olh2012/OpenVLA-Drive
