# Demo UI V3 使用說明

## 🎮 PPO V3 課程學習模型演示工具

`demo_ui_v3.py` 是專為 V3 動態獎勵系統設計的圖形化模型選擇和演示工具。

---

## ✨ 特點

### 1. 專為 V3 設計
- 支援 Stage 1-4 動態獎勵系統
- 正確傳遞 `stage` 參數給環境
- 顯示每個階段的特性說明

### 2. 圖形化界面
- 友好的中文界面
- 兩階段選擇流程：階段 → 模型
- 滑鼠操作，簡單直觀

### 3. 完整信息
- 顯示模型訓練階段
- 顯示棋盤大小
- 顯示獎勵策略類型

---

## 🚀 使用方法

### 快速啟動
```bash
python demo_ui_v3.py
```

### 操作流程

#### 第一步：選擇訓練階段
界面會顯示 4 個階段按鈕：

1. **Stage 1: 6x6 (新手村)**
   - 保守策略
   - 邊緣移動獎勵
   - 耐心覓食

2. **Stage 2: 8x8 (進階班)**
   - 積極策略
   - 主動追食
   - 飢餓懲罰啟動

3. **Stage 3: 10x10 (挑戰班)**
   - 進階策略
   - 強化空間管理
   - 中心開放獎勵

4. **Stage 4: 12x12 (大師班)**
   - 大師策略
   - 平衡策略
   - 最高難度

每個按鈕會顯示該階段有多少個可用模型。

#### 第二步：選擇具體模型
選擇階段後，會列出該階段的所有模型：

- **最佳模型 (best_model.zip)**: 訓練中評估分數最高的模型
- **最終模型 (model.zip)**: 訓練結束時保存的模型

點擊選擇一個模型，然後點擊「開始演示」按鈕。

#### 第三步：觀看演示
- 遊戲會自動運行，AI 自動控制蛇
- 按 **ESC** 可以退出演示
- 每個回合結束後會顯示分數和步數
- 自動開始下一個回合

---

## ⌨️ 快捷鍵

| 按鍵 | 功能 |
|------|------|
| **ESC** | 返回上一頁 / 退出演示 |
| **滑鼠點擊** | 選擇按鈕 |

---

## 📁 模型路徑結構

程序會自動掃描以下目錄：

```
models/ppo_snake_v3_curriculum/
├── Stage1_Novice/
│   ├── best_model/
│   │   └── best_model.zip  ← 最佳模型
│   └── model.zip            ← 最終模型
├── Stage2_Intermediate/
│   ├── best_model/
│   │   └── best_model.zip
│   └── model.zip
├── Stage3_Advanced/
│   ├── best_model/
│   │   └── best_model.zip
│   └── model.zip
└── Stage4_Master/
    ├── best_model/
    │   └── best_model.zip
    └── model.zip
```

---

## 🔧 技術細節

### 環境配置
```python
env = GymSnakeEnvV3(
    board_size=board_size,       # 6, 8, 10, 或 12
    render_mode="human",          # 顯示遊戲畫面
    curriculum_stage=stage_type,  # "conservative" 或 "aggressive"
    stage=stage_num               # 1, 2, 3, 或 4
)
```

### Stage 參數映射
| Stage | Board Size | Curriculum Stage | 策略類型 |
|-------|------------|------------------|----------|
| 1 | 6x6 | conservative | 保守 |
| 2 | 8x8 | aggressive | 積極 |
| 3 | 10x10 | aggressive | 進階 |
| 4 | 12x12 | aggressive | 大師 |

---

## ⚠️ 注意事項

### 1. 需要訓練好的模型
- 運行前必須先完成訓練
- 至少需要一個階段的模型
- 建議先訓練 Stage 1

### 2. 依賴項
```bash
pip install stable-baselines3
pip install pygame
pip install gymnasium
```

### 3. 性能
- 遊戲速度：約 0.05 秒/步
- 可以觀察 AI 的每一步決策
- 按 ESC 隨時退出

---

## 🎯 與舊版本的差異

### `demo_ui.py` (舊版)
- 支援 Q-learning, PPO V1, V2, V3
- 使用舊的 `curriculum_stage` 參數
- 不支援動態 stage 切換

### `demo_ui_v3.py` (新版)
- ✅ 專注於 V3 課程學習模型
- ✅ 支援 `stage=1-4` 參數
- ✅ 正確的動態獎勵系統
- ✅ 階段特性說明
- ✅ 更清晰的界面設計

---

## 🐛 故障排除

### 問題 1: 找不到模型
**症狀**: 所有階段顯示 "0個模型"

**解決方法**:
```bash
# 先訓練至少一個階段
python snake_ai_ppo_v3.py --mode train --stage 1
```

### 問題 2: 模型載入失敗
**症狀**: "模型載入失敗"

**解決方法**:
- 確認模型文件完整（.zip 文件）
- 檢查 stable-baselines3 版本
- 重新訓練該階段

### 問題 3: 中文顯示亂碼
**症狀**: 界面文字顯示方框

**解決方法**:
- Windows 通常自動支援
- 確認系統已安裝中文字體（微軟雅黑等）

---

## 📊 使用範例

### 場景 1: 觀察 Stage 1 學習成果
```bash
python demo_ui_v3.py
# 1. 選擇 "Stage 1: 6x6"
# 2. 選擇 "Stage1_Novice - 最佳模型"
# 3. 觀察 AI 是否靠邊緣移動
```

### 場景 2: 比較不同階段
```bash
# 先觀察 Stage 1
python demo_ui_v3.py  # 選 Stage 1

# 再觀察 Stage 2
python demo_ui_v3.py  # 選 Stage 2

# 對比：
# - Stage 1 應該更多靠邊緣
# - Stage 2 應該更積極追食物
```

### 場景 3: 驗證訓練效果
```bash
# 評估模型
python snake_ai_ppo_v3.py --mode eval --stage 0

# 視覺化觀察
python demo_ui_v3.py  # 選對應階段
```

---

## ✅ 總結

`demo_ui_v3.py` 是專為 V3 動態獎勵系統設計的演示工具：

- ✅ 正確處理 `stage` 參數
- ✅ 支援所有 4 個訓練階段
- ✅ 友好的圖形界面
- ✅ 實時觀察 AI 行為
- ✅ 適合展示和調試

現在您可以輕鬆觀察不同階段的 AI 行為差異！

---

更新時間: 2025-10-26
版本: V3 Dynamic Rewards Compatible
