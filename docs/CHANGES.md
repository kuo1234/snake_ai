# 改動總結 - Snake AI PPO 版本

## 完成的工作

### 1. ✅ 修改計分系統
- **檔案**: `snake_game.py`, `envs/snake_game.py`
- **改動**: 每次吃到食物從 +10 分改為 +1 分
- **最高分**: board_size² - 1
  - 8x8: 最高 63 分
  - 10x10: 最高 99 分

### 2. ✅ 創建 Gymnasium 環境包裝
- **檔案**: `envs/gym_snake_env.py`
- **功能**: 
  - 實現完整的 Gym Env 介面
  - 12 維觀察空間（危險檢測 + 食物方向 + 當前方向）
  - 4 個離散動作（上下左右）
  - 獎勵塑形（reward shaping）
  - 支援 stable_baselines3

### 3. ✅ 創建 PPO 訓練腳本
- **檔案**: `snake_ai_ppo.py`
- **方法**: PPO (Proximal Policy Optimization)
- **框架**: stable_baselines3 + PyTorch
- **功能**:
  - 訓練模式：多進程並行訓練
  - 評估模式：快速性能測試
  - 演示模式：圖形界面觀看 AI
  - 自動保存最佳模型和檢查點
  - Tensorboard 日誌記錄

### 4. ✅ 更新依賴
- **檔案**: `requirements.txt`
- **新增依賴**:
  - gymnasium==0.29.1
  - torch>=2.0.0
  - stable-baselines3>=2.0.0
  - tensorboard>=2.14.0

### 5. ✅ 文檔
- **PPO_README.md**: 詳細的 PPO 使用指南
- **README.md**: 更新主文檔，說明新舊兩種 AI 方法

## 使用方法

### 快速開始

```powershell
# 1. 安裝依賴
pip install -r requirements.txt

# 2. 快速訓練（10萬步，約5-10分鐘）
python snake_ai_ppo.py --mode train --timesteps 100000 --board-size 8

# 3. 觀看 AI 玩遊戲
python snake_ai_ppo.py --mode demo --model-path models/ppo_snake/ppo_snake_final
```

### 推薦訓練流程

1. **初次測試** (10萬步)：
   ```powershell
   python snake_ai_ppo.py --mode train --timesteps 100000 --board-size 6
   ```

2. **標準訓練** (50萬步)：
   ```powershell
   python snake_ai_ppo.py --mode train --timesteps 500000 --board-size 8
   ```

3. **長時間訓練** (100萬步+)：
   ```powershell
   python snake_ai_ppo.py --mode train --timesteps 1000000 --board-size 10
   ```

## 檔案結構

```
snake_ai/
├── envs/
│   ├── __init__.py
│   ├── snake_game.py          # 遊戲核心邏輯（已修改計分）
│   ├── gym_snake_env.py       # Gymnasium 環境包裝（新增）
│   └── config.py
├── snake_ai_ppo.py            # PPO 訓練腳本（新增）
├── snake_ai.py                # Q-learning 方法（原有）
├── train.py                   # Q-learning 訓練（原有）
├── PPO_README.md              # PPO 使用指南（新增）
├── README.md                  # 主文檔（已更新）
└── requirements.txt           # 依賴清單（已更新）
```

## PPO vs Q-learning 比較

| 特性 | PPO | Q-learning |
|-----|-----|-----------|
| 算法類型 | Policy-based (策略梯度) | Value-based (值函數) |
| 神經網路 | 深度神經網路 | 無（使用 Q 表） |
| 訓練速度 | 快（多進程） | 慢（單進程） |
| 擴展性 | 好（適合大狀態空間） | 差（狀態爆炸） |
| 棋盤大小 | 6-16+ | 6-10 |
| 框架 | stable_baselines3 + PyTorch | 自實現 |
| 推薦使用 | ✅ 推薦 | 學習用途 |

## 後續優化方向

1. **調整超參數**:
   - 學習率、batch size、n_steps 等
   - 使用 Optuna 自動調參

2. **改進觀察空間**:
   - 增加更多特徵（蛇身位置、多步前瞻）
   - 使用 CNN 處理圖像輸入

3. **改進獎勵函數**:
   - 更細緻的獎勵塑形
   - 添加中間目標獎勵

4. **嘗試其他算法**:
   - A2C (同步版本的 A3C)
   - DQN (Deep Q-Network)
   - SAC (Soft Actor-Critic)

5. **進階技術**:
   - Curriculum Learning（課程學習）
   - Recurrent Policies（LSTM/GRU）
   - Self-play（自我對弈）

## 測試狀態

✅ 所有核心功能已測試並正常運行：
- Gymnasium 環境創建和運行
- 觀察空間正確（12 維）
- 動作空間正確（4 個動作）
- 計分系統正確（+1 分/食物）
- PPO 腳本可導入並執行

## 開始訓練！

現在你可以開始訓練你的 Snake AI 了！建議從小棋盤（6x6 或 8x8）和較少的訓練步數（10萬）開始，觀察效果後再增加難度。

祝訓練順利！🐍🎮🤖
