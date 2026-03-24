---
name: "lora-finetune"
description: "zi2zi-JiT LoRA微调参数优化指南。用于调整训练参数、数据配置，提升生成效果。召唤场景：用户想优化LoRA训练效果、调整参数或解决训练问题。"
---

# zi2zi-JiT LoRA 微调优化指南

## 1. 当前配置（可调整）

| 参数 | 当前值 | 建议调整范围 | 说明 |
|------|--------|--------------|------|
| EPOCHS | 2000 | 1000-3000 | 训练轮数 |
| MAX_CHARS_PER_FONT | 500 | 500-1000 | 每款字体训练字符数 |
| LORA_R | 32 | 16-64 | LoRA rank（越大拟合能力越强） |
| LORA_ALPHA | 32 | 16-64 | 缩放因子，通常设为 rank 的一半 |
| EARLY_STOP_PATIENCE | 100 | 50-150 | 早停耐心值（epochs） |
| EARLY_STOP_MIN_DELTA | 0.0001 | 0.0001-0.001 | 早停最小改善阈值 |

## 2. 数据配置：Auto-Split 8:2 分配

### 配置参数
```bash
--auto-split \
--train-ratio 0.8 \
--max-chars-per-font 800 \
```

### 工作原理
1. **如果字符数 ≤ 800**：使用所有字符
2. **如果字符数 > 800**：随机选取 800 个字符
3. 选取后按 8:2 比例分配训练集和测试集

### 示例
| 目标字体字符数 | 实际用于训练 | 训练集 | 测试集 |
|---------------|-------------|--------|--------|
| 200 | 200 | 160 | 40 |
| 500 | 500 | 400 | 100 |
| 700 | 700 | 560 | 140 |
| 9000 | 800 | 640 | 160 |

**结论**：无论目标字体有多少字符，最多只使用 800 个字符进行训练，确保训练效率和效果。

## 3. LoRA Rank 详解

### 什么是 LoRA Rank？

LoRA (Low-Rank Adaptation) 通过添加低秩矩阵来微调大模型：

```
原始权重 W (d×k)
LoRA 添加: W' = W + BA
         其中 B (d×r), A (r×k), r = rank
```

### Rank 大小的影响

| Rank 值 | 参数量 | 拟合能力 | 显存占用 | 适用场景 |
|---------|--------|----------|----------|----------|
| 16 | 少 | 较弱 | ~3GB | 快速测试、欠拟合时 |
| 32 | 中等 | 适中（默认） | ~4GB | 常规任务 |
| 64 | 多 | 较强 | ~6GB | 复杂风格 |
| 128 | 很多 | 极强 | ~8GB | 极端风格 |

### 建议
- **默认 32**：适合大多数字体
- **调大到 64**：如果当前效果欠拟合（loss 下降慢）
- **调小到 16**：如果过拟合（训练 loss 低但测试效果差）

## 4. Early Stopping 早停机制

### 参数说明
- **EARLY_STOP_PATIENCE**: 连续多少个 epoch 没有改善就停止
- **EARLY_STOP_MIN_DELTA**: 改善小于此值视为无改善

### 当前配置
```bash
EARLY_STOP_PATIENCE=100    # 100 epochs 无改善则停止
EARLY_STOP_MIN_DELTA=0.0001 # 改善小于 0.0001 视为无改善
```

### 工作原理
1. 每 15 秒检查一次训练 loss
2. 记录最佳 loss
3. 如果连续 100 个 epoch 改善小于 0.0001，自动停止训练
4. 保留最佳 checkpoint

### 调整建议
- **降低 patience**（如 50）：更早停止，适合简单风格
- **增加 patience**（如 150）：更耐心地等待改善，适合复杂风格
- **增大 min_delta**（如 0.001）：更严格的改善标准

## 5. 优化建议：分步执行

### 第一步：当前配置测试
先用当前配置（2000 epochs, rank=32）完成训练，观察 baseline 效果。

### 第二步：增加训练字符
如果效果不满意：
```bash
MAX_CHARS_PER_FONT=800 bash scripts/batch_all_fonts.sh
```

### 第三步：调整 LoRA Rank
如果仍然欠拟合：
```bash
LORA_R=64 bash scripts/batch_all_fonts.sh
```

### 第四步：调整早停
如果训练过早停止：
```bash
EARLY_STOP_PATIENCE=150 bash scripts/batch_all_fonts.sh
```

### 完整参数示例
```bash
MAX_CHARS_PER_FONT=800 \
LORA_R=64 \
LORA_ALPHA=32 \
EPOCHS=2000 \
EARLY_STOP_PATIENCE=150 \
EARLY_STOP_MIN_DELTA=0.0001 \
bash scripts/batch_all_fonts.sh
```

## 6. 训练问题排查

### 问题：训练 loss 下降太慢
- 原因：欠拟合
- 解决：增加 LORA_R 或增加训练字符数

### 问题：训练 loss 低但生成效果差
- 原因：过拟合
- 解决：减少 LORA_R 或减少训练 epochs

### 问题：训练过早停止
- 原因：早停 patience 太小
- 解决：增加 EARLY_STOP_PATIENCE

### 问题：显存不足
- 原因：batch_size 或 LORA_R 太大
- 解决：减少 batch_size 或 LORA_R

## 7. 召唤此 Skill 的场景

- 用户想优化 LoRA 训练效果
- 用户想调整训练参数（epochs、rank、字符数等）
- 用户遇到训练问题（欠拟合、过拟合、早停等）
- 用户想了解某个参数的作用和调整建议
- 用户询问 Auto-Split 数据分配的工作原理
