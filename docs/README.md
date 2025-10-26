# 📚 文檔目錄

這個資料夾包含所有 Snake AI 項目的詳細文檔。

## 📖 文檔列表

### PPO 相關文檔

#### [PPO_README.md](PPO_README.md)
**PPO 快速開始指南**

包含內容：
- 安裝依賴
- 訓練模型（快速/標準/長時間訓練）
- 評估和演示模型
- 模型保存位置
- Tensorboard 監控
- 主要參數說明
- 計分系統
- 訓練建議

適合：想快速開始使用 PPO 訓練 Snake AI 的用戶

---

#### [GPU_SETUP.md](GPU_SETUP.md)
**GPU 訓練設定完整指南** ⚡

包含內容：
- 檢查 GPU 支援和 CUDA 配置
- 安裝 GPU 版本的 PyTorch
- 使用 GPU 訓練（自動/手動配置）
- GPU 訓練效能對比
- 監控 GPU 使用情況
- 最佳化 GPU 訓練（調整環境數、Batch Size、網路架構）
- 常見問題排解（記憶體不足、CUDA 錯誤等）
- 效能測試腳本
- 不同等級 GPU 的建議配置

適合：想使用 GPU 加速訓練的用戶

---

#### [PPO_V2_README.md](PPO_V2_README.md)
**PPO V2 - 增強版碰撞避免訓練** ⭐ 新版本

包含內容：
- V2 主要改進（漸進式碰撞懲罰、身體距離感知、困境偵測）
- 從 12-d 到 16-d 觀察空間
- 更好的獎勵塑形機制
- 訓練和評估指令
- V1 vs V2 效能比較
- 訓練監控和建議
- 技術細節解析

適合：想解決蛇撞到自己身體問題的用戶 🎯

---

#### [PPO_INTRO.md](PPO_INTRO.md)
**PPO 算法詳細介紹**

包含內容：
- 什麼是 PPO？
- 核心概念（Policy Gradient、Clipped Objective、On-Policy Learning）
- PPO 架構（在 Snake 遊戲中的應用）
- 訓練流程圖
- PPO vs Q-learning 比較表
- 觀察空間和獎勵設計
- 訓練參數說明
- 調參建議
- 監控訓練進度（Tensorboard）
- 優勢與局限分析
- 進階優化方向
- 參考資源

適合：想深入了解 PPO 算法原理的用戶

---

### 演示和使用指南

#### [DEMO_GUIDE.md](DEMO_GUIDE.md)
**demo_ai.py 使用指南**

包含內容：
- 功能說明（Q-learning 和 PPO 演示）
- 使用方法（訓練、運行演示、觀看 AI）
- 控制鍵說明
- 輸出資訊解釋
- 性能比較功能
- 故障排除
- 預期效果表格
- 進階使用技巧

適合：想使用 demo_ai.py 觀看訓練結果的用戶

---

### 版本和改動

#### [CHANGES.md](CHANGES.md)
**改動總結和版本歷史**

包含內容：
- 完成的工作清單
- 修改計分系統說明
- Gymnasium 環境創建
- PPO 訓練腳本說明
- 依賴更新記錄
- 文檔創建記錄
- 使用方法總結
- 檔案結構說明
- PPO vs Q-learning 比較
- 後續優化方向

適合：想了解項目更新歷史和改動的用戶

---

## 🚀 快速導航

### 我想...

**開始訓練 PPO 模型**
→ 閱讀 [PPO_README.md](PPO_README.md)

**了解 PPO 算法原理**
→ 閱讀 [PPO_INTRO.md](PPO_INTRO.md)

**觀看訓練好的 AI 玩遊戲**
→ 閱讀 [DEMO_GUIDE.md](DEMO_GUIDE.md)

**查看項目改動歷史**
→ 閱讀 [CHANGES.md](CHANGES.md)

**回到主頁**
→ 返回 [../README.md](../README.md)

---

## 📂 文檔結構

```
docs/
├── README.md           # 本文件（文檔索引）
├── PPO_README.md       # PPO 快速開始
├── PPO_INTRO.md        # PPO 詳細介紹
├── DEMO_GUIDE.md       # demo_ai.py 使用指南
└── CHANGES.md          # 改動總結
```

---

## 🔗 相關鏈接

- [主 README](../README.md) - 項目主頁
- [PPO 訓練腳本](../snake_ai_ppo.py) - 源代碼
- [演示程序](../demo_ai.py) - 源代碼
- [Gymnasium 環境](../envs/gym_snake_env.py) - 環境實現

---

## ❓ 常見問題

**Q: 我應該先讀哪個文檔？**
A: 如果你是新手，建議順序：主 README → PPO_README.md → 開始訓練 → DEMO_GUIDE.md → 觀看效果

**Q: 我想了解算法細節，該讀哪個？**
A: 閱讀 PPO_INTRO.md，裡面有詳細的算法原理和架構說明

**Q: 如何比較 PPO 和 Q-learning？**
A: 運行 `python demo_ai.py` 並選擇選項 5

**Q: 訓練需要多久？**
A: 參考 PPO_README.md 的訓練建議，快速訓練約 5-10 分鐘，標準訓練 30-60 分鐘

---

最後更新：2025-10-25
