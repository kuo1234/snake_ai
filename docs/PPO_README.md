# Snake AI with PPO Quick Start Guide

## 安裝依賴

```powershell
pip install -r requirements.txt
```

## 訓練模型

### 快速訓練（10萬步）
```powershell
python snake_ai_ppo.py --mode train --timesteps 100000 --board-size 8
```

### 標準訓練（50萬步）
```powershell
python snake_ai_ppo.py --mode train --timesteps 500000 --board-size 8
```

### 長時間訓練（100萬步）
```powershell
python snake_ai_ppo.py --mode train --timesteps 1000000 --board-size 10
```

## 評估模型

評估訓練好的模型（不顯示畫面，快速評估）：

```powershell
python snake_ai_ppo.py --mode eval --model-path models/ppo_snake/ppo_snake_final --n-episodes 20
```

## 演示模型

觀看訓練好的 AI 玩遊戲（顯示 pygame 畫面）：

```powershell
python snake_ai_ppo.py --mode demo --model-path models/ppo_snake/ppo_snake_final --n-episodes 5
```

## 模型位置

訓練過程中會自動保存：
- `models/ppo_snake/best_model/best_model.zip` - 評估表現最好的模型
- `models/ppo_snake/ppo_snake_final.zip` - 最終訓練完成的模型
- `models/ppo_snake/ppo_snake_checkpoint_*.zip` - 每 50,000 步的檢查點

## Tensorboard 監控

查看訓練進度和指標：

```powershell
tensorboard --logdir logs/ppo_snake
```

然後在瀏覽器打開 http://localhost:6006

## 主要參數說明

- `--board-size`: 棋盤大小（預設 8，推薦 6-12）
- `--timesteps`: 訓練總步數（預設 100000，推薦至少 500000）
- `--n-envs`: 平行環境數量（預設 4，多核 CPU 可增加）
- `--learning-rate`: 學習率（預設 3e-4）
- `--n-episodes`: 評估/演示的回合數

## 計分系統

- 每吃到一個食物：+1 分
- 最高分：`board_size² - 1`
  - 8x8 棋盤最高分：63 分
  - 10x10 棋盤最高分：99 分
  - 12x12 棋盤最高分：143 分

## 訓練建議

1. **初學者**: 先用 `--board-size 6` 訓練 10 萬步，觀察效果
2. **標準訓練**: 使用 `--board-size 8` 訓練 50-100 萬步
3. **進階訓練**: 使用 `--board-size 10-12` 訓練 100-200 萬步

## 環境特性

- **觀察空間**: 12 維特徵向量
  - 4 個危險檢測（上下左右）
  - 4 個食物方向（上下左右）
  - 4 個當前移動方向（one-hot）
  
- **動作空間**: 4 個離散動作
  - 0: 上 (UP)
  - 1: 左 (LEFT)
  - 2: 右 (RIGHT)
  - 3: 下 (DOWN)

- **獎勵設計**:
  - 吃到食物: +10
  - 撞牆/撞自己: -10
  - 贏得遊戲（填滿棋盤）: +50
  - 每步存活: +0.1
  - 朝食物移動: +0.5
  - 遠離食物: -0.5

## 後續優化方向

未來可以進行的優化：
1. 調整神經網絡架構（增加層數或神經元數量）
2. 調整獎勵函數（reward shaping）
3. 使用更複雜的觀察空間（如視覺輸入）
4. 嘗試其他算法（A2C, DQN, SAC）
5. 添加 curriculum learning（逐步增加難度）
6. 使用 recurrent policies（LSTM）來處理部分可觀察性
