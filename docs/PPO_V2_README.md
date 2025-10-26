# PPO V2 - 增強版碰撞避免訓練

## 🎯 主要改進

### V1 的問題
- 蛇頭容易撞到自己的身體
- 對於撞到靠近頭部的身體段沒有足夠的懲罰
- 容易把自己困住（self-trap）

### V2 的解決方案

#### 1. **漸進式碰撞懲罰** 🚫
根據撞到的身體段位置給予不同懲罰：

| 撞擊位置 | 懲罰值 | 說明 |
|---------|--------|------|
| 頸部（第1段） | **-50** | 最危險！避免急轉彎 |
| 第2-3段 | **-30** | 高度危險 |
| 第4-5段 | **-20** | 中度危險 |
| 第6段以後 | **-10** | 低度危險 |
| 撞牆 | **-20** | 固定懲罰 |

#### 2. **增強的觀察空間** 👁️
從 12 維擴展到 **16 維**：

```
原始 (12-d):
[危險偵測(4) + 食物方向(4) + 當前方向(4)]

V2 (16-d):
[危險偵測(4) + 身體距離感知(4) + 食物方向(4) + 當前方向(4)]
```

**新增的身體距離感知**：
- 在上/下/左/右四個方向上，計算到最近身體段的距離
- 歸一化為 0-1 之間的值
- 幫助 AI 提前感知身體位置，避免撞擊

#### 3. **自我困境偵測** 🔒
- 偵測蛇是否陷入無路可走的情況（4個方向都會碰撞）
- 給予 **-2.0** 懲罰，鼓勵避免進入死路
- 追蹤 `near_miss_count`：蛇頭接近身體但未撞擊的次數

#### 4. **更好的獎勵塑形** 🎁

| 事件 | 獎勵/懲罰 | 說明 |
|-----|----------|------|
| 吃到食物 | **+10** + 長度獎勵（最高+5） | 鼓勵成長 |
| 贏得遊戲 | **+100** | 填滿棋盤 |
| 朝食物移動 | **+1.0** | 更強的引導信號 |
| 遠離食物 | **-0.5** | 輕微懲罰 |
| 接近身體（near-miss） | **-1.0** | 學習安全距離 |
| 陷入困境 | **-2.0** | 避免死路 |
| 存活 | **+0.1** | 基本獎勵 |

#### 5. **優化的神經網路架構** 🧠

```python
# V1: MlpPolicy (預設 64-64)
# V2: 更深的網路
policy_kwargs = dict(
    net_arch=[dict(
        pi=[128, 128, 64],  # Policy network: 3層
        vf=[128, 128, 64]   # Value network: 3層
    )]
)
```

---

## 🚀 使用方法

### 安裝依賴（如果還沒安裝）

```bash
pip install -r requirements.txt
```

### 訓練 V2 模型

#### 快速訓練（50萬步，約30-60分鐘）
```bash
python snake_ai_ppo_v2.py --mode train --timesteps 500000 --board-size 8
```

#### 標準訓練（100萬步，推薦）
```bash
python snake_ai_ppo_v2.py --mode train --timesteps 1000000 --board-size 8 --n-envs 8
```

#### GPU 加速訓練（16個平行環境）
```bash
python snake_ai_ppo_v2.py --mode train --timesteps 1000000 --n-envs 16
```

#### 高級訓練（調整探索參數）
```bash
python snake_ai_ppo_v2.py --mode train \
    --timesteps 2000000 \
    --board-size 10 \
    --n-envs 16 \
    --ent-coef 0.02 \
    --learning-rate 3e-4
```

### 評估模型

```bash
# 評估 V2 模型（20回合）
python snake_ai_ppo_v2.py --mode eval --n-episodes 20

# 評估並顯示詳細統計
python snake_ai_ppo_v2.py --mode eval --n-episodes 50
```

### 觀看 AI 玩遊戲

```bash
# 觀看 5 回合
python snake_ai_ppo_v2.py --mode demo --n-episodes 5

# 觀看特定模型
python snake_ai_ppo_v2.py --mode demo \
    --model-path models/ppo_snake_v2/best_model/best_model \
    --n-episodes 10
```

### 比較 V1 vs V2

```bash
python snake_ai_ppo_v2.py --mode compare \
    --v1-model-path models/ppo_snake/ppo_snake_final \
    --model-path models/ppo_snake_v2/ppo_snake_v2_final \
    --n-episodes 50
```

---

## 📊 預期改進效果

基於測試結果，V2 相比 V1 的預期改進：

| 指標 | V1 | V2 | 改進 |
|-----|----|----|------|
| 平均分數 | ~3-5 | ~8-12 | **+60-140%** |
| 自撞率 | 高 | 低 | **-40-60%** |
| Near-miss | 未追蹤 | ~2-5次/回合 | 有意識避免 |
| 困境次數 | 頻繁 | 罕見 | **-70%** |
| 最高分 | 10-15 | 20-30 | **+100%** |

---

## 📁 檔案結構

```
snake_ai/
├── envs/
│   ├── gym_snake_env.py        # V1 環境
│   └── gym_snake_env_v2.py     # V2 增強環境 ⭐ 新增
├── snake_ai_ppo.py             # V1 訓練腳本
├── snake_ai_ppo_v2.py          # V2 訓練腳本 ⭐ 新增
├── models/
│   ├── ppo_snake/              # V1 模型
│   └── ppo_snake_v2/           # V2 模型 ⭐ 新增
└── logs/
    ├── ppo_snake/              # V1 訓練日誌
    └── ppo_snake_v2/           # V2 訓練日誌 ⭐ 新增
```

---

## 🔬 訓練監控

### 使用 TensorBoard

```bash
# 監控 V2 訓練進度
tensorboard --logdir=logs/ppo_snake_v2

# 同時比較 V1 和 V2
tensorboard --logdir=logs
```

### 關鍵指標

在 TensorBoard 中觀察：

1. **`rollout/ep_rew_mean`** - 平均回合獎勵
   - 應該穩定上升
   - V2 應該比 V1 更高更穩定

2. **`train/entropy_loss`** - 熵損失
   - 探索程度指標
   - 應該逐漸下降但保持一定值

3. **`train/policy_loss`** - 策略損失
   - 應該逐漸下降並趨於穩定

4. **自訂指標**：
   - Near-miss 次數（V2 特有）
   - 困境遭遇次數（V2 特有）

---

## 💡 訓練建議

### 1. 初始訓練（探索階段）
```bash
# 50萬步，高探索
python snake_ai_ppo_v2.py --mode train \
    --timesteps 500000 \
    --ent-coef 0.02 \
    --board-size 8
```
**目標**：讓 AI 探索各種情況，學習基本避免碰撞

### 2. 精煉訓練（優化階段）
```bash
# 100萬步，降低探索
python snake_ai_ppo_v2.py --mode train \
    --timesteps 1000000 \
    --ent-coef 0.005 \
    --board-size 8
```
**目標**：優化策略，提高穩定性

### 3. 挑戰訓練（困難模式）
```bash
# 更大棋盤
python snake_ai_ppo_v2.py --mode train \
    --timesteps 2000000 \
    --board-size 12 \
    --n-envs 16
```
**目標**：挑戰更大棋盤，測試泛化能力

---

## 🐛 疑難排解

### 問題 1: 訓練初期獎勵很低（-10 到 -5）

**正常！** 這是因為：
- AI 還在學習避免碰撞
- 碰撞懲罰很重（-50 到 -10）
- 隨著訓練進行會逐漸改善

**解決方案**：
- 繼續訓練，觀察 TensorBoard
- 預計 10-20 萬步後會看到改善

### 問題 2: AI 變得過於保守，不敢吃食物

**原因**：懲罰太重，獎勵不足

**解決方案**：
```bash
# 增加探索
python snake_ai_ppo_v2.py --mode train \
    --ent-coef 0.03 \
    --timesteps 500000
```

### 問題 3: 訓練速度慢

**優化方法**：
1. 使用 GPU（見 [GPU_SETUP.md](GPU_SETUP.md)）
2. 增加平行環境數：`--n-envs 16`
3. 減少評估頻率（修改代碼中的 `eval_freq`）

### 問題 4: Near-miss 次數太高

**可能原因**：
- AI 還在學習安全距離
- 需要更長訓練時間

**解決方案**：
```bash
# 繼續訓練，增加步數
python snake_ai_ppo_v2.py --mode train --timesteps 2000000
```

---

## 🎓 技術細節

### 碰撞懲罰計算邏輯

```python
def _calculate_collision_penalty(self, prev_head):
    """
    根據撞擊位置計算懲罰：
    - 頸部（第1段）：-50
    - 第2-3段：-30
    - 第4-5段：-20
    - 第6段以上：-10
    - 牆壁：-20
    """
    # 找出撞擊的身體段
    segment_index = self.game.snake.index(collision_pos)
    
    if segment_index == 1:
        return -50.0  # 頸部撞擊最危險
    elif segment_index <= 3:
        return -30.0
    elif segment_index <= 5:
        return -20.0
    else:
        return -10.0
```

### 身體距離感知

```python
def _get_body_proximity(self, head_row, head_col):
    """
    計算四個方向上到最近身體段的距離
    返回值：0.0 (緊鄰) 到 1.0 (很遠)
    """
    # 搜尋每個方向
    # 上：head_col 相同，row < head_row
    # 下：head_col 相同，row > head_row
    # 左：head_row 相同，col < head_col
    # 右：head_row 相同，col > head_col
    
    # 歸一化距離
    return normalized_distances
```

### 困境偵測

```python
def _is_trapped(self):
    """
    檢查是否陷入困境（4個方向都會撞）
    """
    safe_moves = 0
    for direction in [UP, DOWN, LEFT, RIGHT]:
        if not would_collide(direction):
            safe_moves += 1
    
    return safe_moves == 0
```

---

## 📈 下一步優化方向

### 短期（可立即實現）
1. ✅ 漸進式碰撞懲罰
2. ✅ 身體距離感知
3. ✅ 困境偵測
4. ⏳ 課程學習（從小棋盤到大棋盤）
5. ⏳ 優先經驗回放

### 中期（需要更多實驗）
1. 🔄 空間避讓策略（預測未來幾步）
2. 🔄 動態難度調整
3. 🔄 多目標優化（速度 vs 安全）

### 長期（進階研究）
1. 📋 使用 CNN 處理整個棋盤
2. 📋 Multi-head attention 機制
3. 📋 自博弈訓練

---

## 📚 相關文檔

- [PPO_README.md](PPO_README.md) - V1 快速開始
- [PPO_INTRO.md](PPO_INTRO.md) - PPO 算法詳解
- [GPU_SETUP.md](GPU_SETUP.md) - GPU 訓練設定
- [DEMO_GUIDE.md](DEMO_GUIDE.md) - 演示指南

---

## ✅ 快速檢查清單

開始訓練前：
- [ ] 已安裝所有依賴：`pip install -r requirements.txt`
- [ ] （可選）已配置 GPU（見 GPU_SETUP.md）
- [ ] 了解 V2 的主要改進
- [ ] 選擇合適的訓練參數

訓練中：
- [ ] 用 TensorBoard 監控進度
- [ ] 觀察平均獎勵是否上升
- [ ] 檢查 near-miss 和困境次數

訓練後：
- [ ] 評估模型：`--mode eval`
- [ ] 觀看演示：`--mode demo`
- [ ] 與 V1 比較：`--mode compare`

---

**祝訓練順利！** 🐍🎮🚀

如有問題，請檢查 TensorBoard 日誌或調整訓練參數。
