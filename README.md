# Snake Game with AI

一個使用 pygame 制作的經典貪吃蛇遊戲，結合了多種強化學習算法的人工智慧。

## 功能特點

### 🎮 經典遊戲模式
- 經典的貪吃蛇遊戲玩法
- 美觀的圖形界面，包括：
  - 鑽石形狀的蛇頭帶眼睛
  - 漸變色的蛇身
  - 紅色的食物
- 音效支持（可選）
- 歡迎界面和遊戲結束界面
- 分數顯示和倒計時開始

### 🤖 AI智能體

#### 1. PPO V3 (Curriculum Learning) - 課程學習版本 🎓⭐⭐
- **最新創新**: 引入課程學習（Curriculum Learning）
- **階段式訓練**: 從 6x6 → 8x8 → 10x10 → 12x12 循序漸進
- **遷移學習**: 前一階段知識遷移到下一階段
- **更強網路**: [256, 256, 128] 神經網路
- **自動化流程**: 自動檢查畢業標準並升級
- **大棋盤突破**: 在 10x10, 12x12 上表現優異
- **詳細文檔**: 見 [docs/PPO_V3_CURRICULUM_LEARNING.md](docs/PPO_V3_CURRICULUM_LEARNING.md)

**快速開始 PPO V3**:
```bash
# 完整課程訓練（4個階段，總計約 6-9 小時）
python snake_ai_ppo_v3.py --mode train --device auto --n-envs 8

# 使用 GPU 加速課程訓練
python snake_ai_ppo_v3.py --mode train --device cuda --n-envs 16

# 觀看特定階段的 AI（0=6x6, 1=8x8, 2=10x10, 3=12x12）
python snake_ai_ppo_v3.py --mode demo --stage 2

# 評估課程學習效果
python snake_ai_ppo_v3.py --mode eval --stage 3 --n-episodes 50
```

#### 2. PPO V2 (Enhanced Collision Avoidance) - 微觀教練版本 ⭐🌟
- **微觀教練**: 漸進式懲罰，教導避免自撞
- **增強觀察空間**: 16-d（包含身體距離感知）
- **困境偵測**: 避免陷入無路可走的情況
- **更好的獎勵塑形**: 學習安全導航
- **詳細文檔**: 見 [docs/PPO_V2_README.md](docs/PPO_V2_README.md)

**快速開始 PPO V2**:
```bash
# 訓練 V2（50萬步，約30-60分鐘）
python snake_ai_ppo_v2.py --mode train --timesteps 500000 --board-size 8

# 使用 GPU 加速（16個平行環境）
python snake_ai_ppo_v2.py --mode train --timesteps 1000000 --n-envs 16

# 觀看 V2 AI 玩遊戲
python snake_ai_ppo_v2.py --mode demo --n-episodes 5

# 比較 V1 vs V2 效能
python snake_ai_ppo_v2.py --mode compare --n-episodes 50
```

#### 3. PPO V1 (Proximal Policy Optimization) - 基礎版本 🌟
- **最新方法**: 使用 stable_baselines3 和 PyTorch
- **高效訓練**: 支持多進程並行訓練
- **現代架構**: 深度神經網路策略
- **易於使用**: 簡單的命令行介面
- **詳細文檔**: 見 [docs/PPO_README.md](docs/PPO_README.md)

**快速開始 PPO**:
```bash
# 安裝依賴（包含 torch, stable-baselines3, gymnasium）
pip install -r requirements.txt

# 訓練 AI（10萬步快速訓練）
python snake_ai_ppo.py --mode train --timesteps 100000 --board-size 8

# 使用 GPU 加速訓練（自動偵測）⚡
python snake_ai_ppo.py --mode train --timesteps 500000 --n-envs 16

# 觀看訓練好的 AI 玩遊戲
python snake_ai_ppo.py --mode demo --model-path models/ppo_snake/ppo_snake_final
```

💡 **GPU 加速**: 見 [docs/GPU_SETUP.md](docs/GPU_SETUP.md) 了解如何配置和使用 GPU 訓練

#### 4. Q-learning（傳統方法）
- **傳統算法**: 基於貝爾曼方程的強化學習
- **狀態簡化**: 將複雜遊戲狀態壓縮為12個關鍵特徵
- **Q表學習**: 使用表格方法存儲狀態-動作值
- **適合小棋盤**: 在較小的棋盤（6x6, 8x8）上表現良好

## 計分系統

- 每吃到一個食物：**+1 分**
- 最高分：**board_size² - 1**
  - 8x8 棋盤：最高 63 分
  - 10x10 棋盤：最高 99 分
  - 12x12 棋盤：最高 143 分

## 安裝要求

### 基礎依賴
- Python 3.8+
- pygame
- numpy  
- matplotlib

### PPO 額外依賴
- torch (PyTorch)
- stable-baselines3
- gymnasium
- tensorboard

## 安裝

1. 安裝所有依賴包：
```bash
pip install -r requirements.txt
```

## 運行方式

### 🎨 1. 圖形化模型選擇器（最簡單，推薦新手）⭐

**一鍵啟動，滑鼠點擊選擇模型和設定！**

```bash
# 方法 1：批次檔（Windows）
run_ui_selector.bat

# 方法 2：Python 腳本
python demo_ui.py

# 方法 3：從主選單啟動
python demo_ai.py  # 然後選擇選項 0
```

**功能特點：**
- 🖱️ 滑鼠點擊選擇，無需命令行
- 📊 自動掃描所有可用模型
- 🎯 可選擇 Q-learning、PPO V1、PPO V2
- 📏 可選擇棋盤大小（6x6、8x8、10x10、12x12）
- ✨ 美觀的 pygame UI 界面

詳細使用指南請見 [docs/UI_SELECTOR_GUIDE.md](docs/UI_SELECTOR_GUIDE.md)

---

### 2. 人類玩家模式
```bash
python snake_game.py
```

---

### 3. PPO AI 訓練（推薦）
```bash
# 快速訓練（10萬步，約5-10分鐘）
python snake_ai_ppo.py --mode train --timesteps 100000 --board-size 8

# 標準訓練（50萬步，約30-60分鐘）
python snake_ai_ppo.py --mode train --timesteps 500000 --board-size 8

# 評估模型
python snake_ai_ppo.py --mode eval --n-episodes 20

# 觀看 AI 玩遊戲（有圖形界面）
python snake_ai_ppo.py --mode demo --n-episodes 5
```

詳細的 PPO 使用說明請見 [docs/PPO_README.md](docs/PPO_README.md)

---

### 4. Q-learning AI 訓練（傳統方法）
```bash
# 快速訓練 (500回合, 6x6棋盤)
python train.py --mode quick

# 標準訓練 (2000回合, 8x8棋盤)  
python train.py --mode standard

# 深度訓練 (5000回合, 10x10棋盤)
python train.py --mode intensive

# 參數比較實驗
python train.py --mode compare
```

---

### 5. AI演示模式（命令行選單）

安裝 gymnasium（已加入到 `requirements.txt`）：

```bash
pip install -r requirements.txt
```

使用隨機策略運行示例：

```bash
python demo_gym.py
```

注意：這個封裝目前提供二進位特徵觀察（12 維），與 `SnakeAI` 使用的狀態表示一致，方便快速集成自定義訓練循環或第三方 RL 庫。

## 遊戲控制

### 人類玩家
- **方向鍵** ↑↓←→ - 控制蛇的移動方向
- **滑鼠點擊** - 在歡迎界面點擊START開始遊戲，在遊戲結束界面點擊RETRY重新開始

### AI觀察模式
- **ESC鍵** - 退出AI演示
- **空格鍵** - 暫停/繼續AI遊戲

## 遊戲規則

1. 控制蛇吃紅色的食物
2. 每吃一個食物得10分，蛇身長度增加1  
3. 撞到牆壁或撞到自己的身體遊戲結束
4. 目標是獲得盡可能高的分數

## AI算法原理

### 貝爾曼方程 (Bellman Equation)
```
Q(s,a) = R(s,a) + γ * max(Q(s',a'))
```

我們的AI使用Q-learning算法，其中：
- **狀態 (s)**: 12維特徵向量（危險檢測 + 食物方向 + 當前方向）
- **動作 (a)**: 上下左右四個方向
- **獎勵 (R)**: 吃食物+50分，存活+1分，死亡-100分
- **折扣因子 (γ)**: 0.95（重視長期獎勵）

### 學習過程
1. **探索階段**: 隨機嘗試不同動作，建立Q表
2. **學習階段**: 根據貝爾曼方程更新Q值
3. **利用階段**: 選擇Q值最高的動作

詳細說明請參考 `BELLMAN_EQUATION.md`

## 項目結構

- `snake_game.py` - 主遊戲文件
- `snake_ai.py` - AI智能體（Q-learning算法）
- `train.py` - AI訓練腳本  
- `demo_ai.py` - AI演示程序
- `demo.py` - 遊戲演示腳本
- `config.py` - 遊戲配置文件
- `requirements.txt` - 依賴包列表
- `BELLMAN_EQUATION.md` - 貝爾曼方程詳細說明
- `sound/` - 音效文件目錄（可選）

## AI訓練結果

我們的實驗結果顯示：

| 模式 | 棋盤大小 | 訓練回合 | 平均分數 | 學習效果 |
|------|---------|---------|---------|---------|
| 快速 | 6×6     | 500     | 90分    | 基礎策略 |
| 標準 | 8×8     | 2000    | 160分   | 優秀策略 |
| 深度 | 10×10   | 5000    | >300分  | 專家級別 |

## 可選音效

如果想要添加音效，請在項目根目錄創建 `sound` 資料夾，並放入以下文件：
- `eat.wav` - 吃食物音效
- `game_over.wav` - 遊戲結束音效
- `victory.wav` - 勝利音效

## 自定義選項

遊戲支援以下自定義選項：

### SnakeGame 參數
- `board_size` - 遊戲板大小（預設12×12）
- `seed` - 隨機種子
- `silent_mode` - 靜默模式（不顯示圖形界面）

### SnakeAI 參數  
- `learning_rate` - 學習率（預設0.1）
- `discount_factor` - 折扣因子（預設0.95）
- `epsilon` - 初始探索率（預設1.0）
- `epsilon_decay` - 探索衰減率（預設0.995）

## 📚 文檔索引

所有詳細文檔都存放在 [`docs/`](docs/) 資料夾中：

- **[PPO_V2_README.md](docs/PPO_V2_README.md)** - PPO V2 增強版訓練指南（解決碰撞問題）⭐ 推薦
- **[PPO_README.md](docs/PPO_README.md)** - PPO 訓練快速開始指南
- **[GPU_SETUP.md](docs/GPU_SETUP.md)** - GPU 訓練設定完整指南 ⚡
- **[PPO_INTRO.md](docs/PPO_INTRO.md)** - PPO 算法詳細介紹（原理、架構、訓練建議）
- **[DEMO_GUIDE.md](docs/DEMO_GUIDE.md)** - demo_ai.py 使用指南
- **[CHANGES.md](docs/CHANGES.md)** - 改動總結和版本歷史

## 未來改進方向

1. **深度Q網路 (DQN)**: 使用神經網路代替Q表
2. **卷積神經網路**: 直接處理遊戲畫面
3. **多智能體訓練**: 競爭性學習環境  
4. **優先經驗回放**: 提高學習效率
5. **Double Q-learning**: 減少Q值過估計問題