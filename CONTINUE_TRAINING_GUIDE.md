# 从已训练模型继续训练指南

## 🎯 场景：Stage 1 已完成，从 Stage 2 开始

如果您的 Stage 1 (6x6) 已经训练完成并且表现良好，不需要重新训练，可以直接从 Stage 2 开始。

---

## 🚀 方法一：使用快速启动脚本（推荐）

### 1. 运行快速脚本
```bash
python train_from_stage2.py
```

### 2. 脚本会自动：
- ✓ 检查 Stage 1 模型是否存在
- ✓ 验证模型路径
- ✓ 从 Stage 2 开始训练
- ✓ 自动加载 Stage 1 模型进行迁移学习

---

## 🔧 方法二：使用命令行参数

### 从 Stage 2 开始
```bash
python snake_ai_ppo_v3.py --mode train --start-stage 1
```
**注意**: `--start-stage 1` 表示从第 2 个阶段开始（索引从 0 开始）

### 从 Stage 3 开始
```bash
python snake_ai_ppo_v3.py --mode train --start-stage 2
```

### 从 Stage 4 开始
```bash
python snake_ai_ppo_v3.py --mode train --start-stage 3
```

---

## 📋 阶段索引对照表

| 阶段名称 | 棋盘大小 | --start-stage 参数 |
|---------|----------|-------------------|
| Stage 1 (新手村) | 6x6 | 0 |
| Stage 2 (进阶班) | 8x8 | 1 |
| Stage 3 (挑战班) | 10x10 | 2 |
| Stage 4 (大师班) | 12x12 | 3 |

---

## 🔍 系统会自动检测

当您使用 `--start-stage N` (N > 0) 时，系统会：

### 1. 自动查找前一阶段模型
```
查找路径: models/ppo_snake_v3_curriculum/Stage{N}_*/best_model/best_model.zip
```

### 2. 加载成功
```
🔍 發現已訓練的 Stage1_Novice 模型
   路徑: models/ppo_snake_v3_curriculum/Stage1_Novice/best_model/best_model.zip
   ✓ 成功載入前一階段模型，將用於遷移學習
```

### 3. 应用增强参数震荡
```
⚡ 強化參數震盪: 提高學習率 + 探索率以打破舊策略...
   - 震盪階段: LR = 3e-4, Entropy = 0.02
   - 訓練 150,000 步（強制重新探索）
   
   - 過渡階段: LR = 1.5e-4, Entropy = 0.015  
   - 訓練 100,000 步（穩定策略）
   
   - 穩定階段: LR = 3e-4, Entropy = 0.01
   - 繼續訓練...
```

### 4. 如果找不到模型
```
⚠️  警告: 找不到 Stage1_Novice 的模型
   預期路徑: models/ppo_snake_v3_curriculum/Stage1_Novice/best_model/best_model.zip
   建議先訓練 Stage 1
   是否繼續從頭訓練 Stage 2? (y/n):
```

---

## 💡 实用场景

### 场景 1: Stage 1 已完美，直接训练 Stage 2
```bash
# 检查 Stage 1 模型存在
python demo_ui_v3.py  # 选择 Stage 1 观察效果

# 确认后开始 Stage 2 训练
python train_from_stage2.py
```

### 场景 2: Stage 1-2 已完成，训练 Stage 3
```bash
python snake_ai_ppo_v3.py --mode train --start-stage 2
```

### 场景 3: 重新训练某个阶段
```bash
# 如果 Stage 2 效果不好，想重新训练
# 但保留 Stage 1 的模型

# 删除 Stage 2 模型
rm -r models/ppo_snake_v3_curriculum/Stage2_Intermediate

# 重新从 Stage 2 开始
python train_from_stage2.py
```

---

## 🎓 训练流程建议

### 完整训练流程（从零开始）
```bash
# Step 1: 训练 Stage 1
python snake_ai_ppo_v3.py --mode train --start-stage 0

# Step 2: 评估 Stage 1
python snake_ai_ppo_v3.py --mode eval --stage 0

# Step 3: 观察 Stage 1
python demo_ui_v3.py  # 选择 Stage 1

# Step 4: 如果 Stage 1 满意，训练 Stage 2
python train_from_stage2.py

# Step 5: 评估 Stage 2
python snake_ai_ppo_v3.py --mode eval --stage 1

# ... 重复 Stage 3, 4
```

### 快速迭代流程（Stage 1 已完成）
```bash
# 直接训练 Stage 2-4
python train_from_stage2.py

# 系统会自动：
# 1. 加载 Stage 1 → 训练 Stage 2
# 2. 加载 Stage 2 → 训练 Stage 3  
# 3. 加载 Stage 3 → 训练 Stage 4
```

---

## ⚙️ 新增的增强功能

### 1. 智能模型检测
- 自动查找前一阶段的最佳模型
- 验证模型文件完整性
- 提供友好的错误提示

### 2. 增强参数震荡（对抗卡死）
当从 Stage 1 迁移到 Stage 2+ 时：

**震荡阶段** (150k 步):
- 学习率: 3e-4 (高)
- 探索率: 0.02 (高)
- 目的: 打破旧策略，强制探索

**过渡阶段** (100k 步):
- 学习率: 1.5e-4 (中)
- 探索率: 0.015 (中)  
- 目的: 稳定新策略

**稳定阶段** (250k 步):
- 学习率: 3e-4 (正常)
- 探索率: 0.01 (正常)
- 目的: 精细优化

### 3. 大幅提高的陷阱惩罚
- Stage 2: -10.0 (原 -3.0)
- Stage 3: -15.0 (原 -5.0)
- Stage 4: -15.0 (原 -5.0)

目的: 让 AI 重新害怕卡死

---

## 📁 模型文件结构

```
models/ppo_snake_v3_curriculum/
├── Stage1_Novice/              ← 已完成
│   ├── best_model/
│   │   └── best_model.zip      ← 系统会自动加载这个
│   ├── model.zip
│   └── logs/
├── Stage2_Intermediate/         ← 正在训练
│   ├── best_model/             ← 训练中会保存最佳模型
│   ├── checkpoints/            ← 每 50k 步保存
│   └── logs/
├── Stage3_Advanced/            ← 等待训练
└── Stage4_Master/              ← 等待训练
```

---

## ✅ 检查清单

开始从 Stage 2 训练前，确认：

- [ ] Stage 1 模型存在
  ```bash
  ls models/ppo_snake_v3_curriculum/Stage1_Novice/best_model/best_model.zip
  ```

- [ ] Stage 1 评估分数 ≥ 25
  ```bash
  python snake_ai_ppo_v3.py --mode eval --stage 0
  ```

- [ ] Stage 1 表现满意（观察演示）
  ```bash
  python demo_ui_v3.py  # 选择 Stage 1
  ```

- [ ] 磁盘空间充足（约 500MB）

- [ ] GPU 可用（可选但推荐）
  ```bash
  python -c "import torch; print(torch.cuda.is_available())"
  ```

---

## 🐛 常见问题

### Q1: 找不到 Stage 1 模型
**A**: 检查文件路径是否正确：
```bash
# 应该存在以下任一文件
ls models/ppo_snake_v3_curriculum/Stage1_Novice/best_model/best_model.zip
ls models/ppo_snake_v3_curriculum/Stage1_Novice/model.zip
```

### Q2: Stage 2 还是会卡死
**A**: 
1. 确认陷阱惩罚已更新为 -10.0
2. 检查参数震荡是否执行（训练开始时会显示）
3. 增加震荡步数或提高探索率

### Q3: 训练速度太慢
**A**: 
1. 使用 GPU: `--device cuda`
2. 减少并行环境: `--n-envs 4`
3. 使用更快的 CPU

---

## 🎯 总结

现在您可以：

✅ **跳过已完成的阶段**
```bash
python train_from_stage2.py
```

✅ **自动加载前一阶段模型**
- 系统会查找并加载 Stage 1 模型
- 应用增强参数震荡
- 大幅提高陷阱惩罚

✅ **灵活的训练控制**
- 从任何阶段开始
- 重新训练特定阶段
- 保留已完成的训练成果

开始训练 Stage 2，解决卡死问题！🚀

---

更新时间: 2025-10-26
版本: V3 with Skip Completed Stages
