# Snake Game with AI

一個使用 pygame 制作的經典貪吃蛇遊戲，結合了基於貝爾曼方程的Q-learning人工智慧。

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
- **Q-learning算法**: 基於貝爾曼方程的強化學習
- **狀態簡化**: 將複雜遊戲狀態壓縮為12個關鍵特徵
- **智能決策**: AI學會避開危險並高效收集食物
- **訓練可視化**: 實時查看訓練進度和AI表現

## 安裝要求

- Python 3.7+
- pygame
- numpy  
- matplotlib

## 安裝

1. 安裝依賴包：
```bash
pip install -r requirements.txt
```

## 運行方式

### 1. 人類玩家模式
```bash
python snake_game.py
```

### 2. AI訓練模式
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

### 3. AI演示模式
```bash
# 觀看AI遊戲 (需要已訓練的模型)
python demo_ai.py

# 或使用訓練腳本的演示模式
python train.py --mode demo
```

### 4. 模型分析
```bash
# 分析Q表內容
python train.py --mode analyze --model snake_ai_standard.pkl
```

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

## 未來改進方向

1. **深度Q網路 (DQN)**: 使用神經網路代替Q表
2. **卷積神經網路**: 直接處理遊戲畫面
3. **多智能體訓練**: 競爭性學習環境  
4. **優先經驗回放**: 提高學習效率
5. **Double Q-learning**: 減少Q值過估計問題