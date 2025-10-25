import numpy as np
import random
import pickle
import matplotlib.pyplot as plt
from collections import deque
from envs.snake_game import SnakeGame

class SnakeAI:
    """
    使用Q-learning算法訓練的貪吃蛇AI
    基於貝爾曼方程: Q(s,a) = R(s,a) + γ * max(Q(s',a'))
    """
    
    def __init__(self, board_size=12, learning_rate=0.1, discount_factor=0.95, 
                 epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
        self.board_size = board_size
        self.learning_rate = learning_rate  # α - 學習率
        self.discount_factor = discount_factor  # γ - 折扣因子
        self.epsilon = epsilon  # ε - 探索率
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        # 動作空間: 0=UP, 1=LEFT, 2=RIGHT, 3=DOWN
        self.actions = [0, 1, 2, 3]
        self.action_size = len(self.actions)
        
        # 狀態特徵數量 (簡化的狀態表示)
        self.state_size = 12  # 危險檢測 + 食物方向 + 當前方向
        
        # Q表: 狀態 -> 動作值
        self.q_table = {}
        
        # 訓練統計
        self.scores = []
        self.episodes = 0
        
    def get_state(self, game):
        """
        獲取當前遊戲狀態的特徵表示
        簡化狀態包括:
        1. 危險檢測 (上下左右是否有危險)
        2. 食物相對位置
        3. 當前移動方向
        """
        if not game.snake:
            return tuple([0] * self.state_size)
            
        head_row, head_col = game.snake[0]
        food_row, food_col = game.food
        
        # 1. 危險檢測 (4個方向)
        danger_up = (head_row - 1 < 0 or 
                    (head_row - 1, head_col) in game.snake)
        danger_down = (head_row + 1 >= self.board_size or 
                      (head_row + 1, head_col) in game.snake)
        danger_left = (head_col - 1 < 0 or 
                      (head_row, head_col - 1) in game.snake)
        danger_right = (head_col + 1 >= self.board_size or 
                       (head_row, head_col + 1) in game.snake)
        
        # 2. 食物方向 (4個方向)
        food_up = food_row < head_row
        food_down = food_row > head_row
        food_left = food_col < head_col
        food_right = food_col > head_col
        
        # 3. 當前方向 (4個方向的one-hot編碼)
        dir_up = game.direction == "UP"
        dir_down = game.direction == "DOWN"
        dir_left = game.direction == "LEFT"
        dir_right = game.direction == "RIGHT"
        
        state = (
            int(danger_up), int(danger_down), int(danger_left), int(danger_right),
            int(food_up), int(food_down), int(food_left), int(food_right),
            int(dir_up), int(dir_down), int(dir_left), int(dir_right)
        )
        
        return state
    
    def get_q_values(self, state):
        """獲取狀態的Q值"""
        if state not in self.q_table:
            # 初始化新狀態的Q值為0
            self.q_table[state] = np.zeros(self.action_size)
        return self.q_table[state]
    
    def choose_action(self, state, training=True):
        """
        使用ε-貪心策略選擇動作
        """
        if training and np.random.random() <= self.epsilon:
            # 探索: 隨機選擇動作
            return random.choice(self.actions)
        else:
            # 利用: 選擇Q值最高的動作
            q_values = self.get_q_values(state)
            return np.argmax(q_values)
    
    def update_q_table(self, state, action, reward, next_state, done):
        """
        使用貝爾曼方程更新Q表
        Q(s,a) = Q(s,a) + α * [R + γ * max(Q(s',a')) - Q(s,a)]
        """
        current_q = self.get_q_values(state)[action]
        
        if done:
            # 遊戲結束，沒有未來獎勵
            target_q = reward
        else:
            # 計算目標Q值: R + γ * max(Q(s',a'))
            next_q_values = self.get_q_values(next_state)
            target_q = reward + self.discount_factor * np.max(next_q_values)
        
        # 貝爾曼方程更新
        new_q = current_q + self.learning_rate * (target_q - current_q)
        self.q_table[state][action] = new_q
    
    def calculate_reward(self, game, done, info):
        """
        計算獎勵函數
        """
        reward = 0
        
        if done:
            if len(game.snake) == game.grid_size:
                # 贏得遊戲 (填滿整個板子)
                reward = 100
            else:
                # 遊戲結束 (撞牆或撞到自己)
                reward = -100
        elif info['food_obtained']:
            # 吃到食物
            reward = 50
        else:
            # 每步存活的小獎勵
            reward = 1
            
            # 額外獎勵: 朝食物方向移動
            head_pos = info['snake_head_pos']
            food_pos = info['food_pos']
            prev_distance = np.abs(info['prev_snake_head_pos'] - food_pos).sum()
            current_distance = np.abs(head_pos - food_pos).sum()
            
            if current_distance < prev_distance:
                reward += 2  # 接近食物
            elif current_distance > prev_distance:
                reward -= 1  # 遠離食物
        
        return reward
    
    def train_episode(self, game):
        """
        訓練一個回合
        """
        game.reset()
        state = self.get_state(game)
        total_reward = 0
        steps = 0
        max_steps = self.board_size * self.board_size * 2  # 防止無限循環
        
        while steps < max_steps:
            # 選擇動作
            action = self.choose_action(state, training=True)
            
            # 執行動作
            done, info = game.step(action)
            
            # 獲取新狀態和獎勵
            next_state = self.get_state(game)
            reward = self.calculate_reward(game, done, info)
            
            # 更新Q表 (貝爾曼方程)
            self.update_q_table(state, action, reward, next_state, done)
            
            total_reward += reward
            state = next_state
            steps += 1
            
            if done:
                break
        
        # 記錄統計信息
        self.scores.append(game.score)
        self.episodes += 1
        
        # 衰減探索率
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        return total_reward, game.score, steps
    
    def train(self, num_episodes=1000, save_every=100, verbose=True):
        """
        訓練AI
        """
        game = SnakeGame(silent_mode=True, board_size=self.board_size)
        
        print(f"開始訓練 {num_episodes} 回合...")
        print("使用貝爾曼方程進行Q-learning")
        
        for episode in range(num_episodes):
            total_reward, score, steps = self.train_episode(game)
            
            if verbose and episode % 100 == 0:
                avg_score = np.mean(self.scores[-100:]) if len(self.scores) >= 100 else np.mean(self.scores)
                print(f"Episode {episode}: Score={score}, Avg Score={avg_score:.2f}, "
                      f"Steps={steps}, Epsilon={self.epsilon:.3f}, Q-table size={len(self.q_table)}")
            
            # 定期保存模型
            if save_every > 0 and episode % save_every == 0 and episode > 0:
                self.save_model(f"snake_ai_episode_{episode}.pkl")
        
        print("訓練完成！")
        return self.scores
    
    def play_game(self, game=None, render=True):
        """
        讓AI玩一局遊戲
        """
        if game is None:
            game = SnakeGame(silent_mode=not render, board_size=self.board_size)
        
        game.reset()
        state = self.get_state(game)
        steps = 0
        max_steps = self.board_size * self.board_size * 2
        
        while steps < max_steps:
            # AI選擇動作 (不探索，只利用)
            action = self.choose_action(state, training=False)
            
            # 執行動作
            done, info = game.step(action)
            
            if render and hasattr(game, 'render'):
                game.render()
                import time
                time.sleep(0.1)  # 延遲以便觀看
            
            if done:
                break
                
            state = self.get_state(game)
            steps += 1
        
        return game.score, steps
    
    def save_model(self, filename):
        """保存訓練好的模型"""
        model_data = {
            'q_table': self.q_table,
            'epsilon': self.epsilon,
            'episodes': self.episodes,
            'scores': self.scores,
            'board_size': self.board_size,
            'learning_rate': self.learning_rate,
            'discount_factor': self.discount_factor
        }
        
        with open(filename, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"模型已保存到 {filename}")
    
    def load_model(self, filename):
        """載入訓練好的模型"""
        try:
            with open(filename, 'rb') as f:
                model_data = pickle.load(f)
            
            self.q_table = model_data['q_table']
            self.epsilon = model_data.get('epsilon', self.epsilon_min)
            self.episodes = model_data.get('episodes', 0)
            self.scores = model_data.get('scores', [])
            
            print(f"模型已從 {filename} 載入")
            print(f"Q表大小: {len(self.q_table)}, 訓練回合: {self.episodes}")
            
        except FileNotFoundError:
            print(f"找不到文件 {filename}")
        except Exception as e:
            print(f"載入模型時出錯: {e}")
    
    def plot_training_progress(self):
        """繪製訓練進度"""
        if not self.scores:
            print("沒有訓練數據可繪製")
            return
        
        plt.figure(figsize=(12, 4))
        
        # 分數曲線
        plt.subplot(1, 2, 1)
        plt.plot(self.scores)
        if len(self.scores) >= 100:
            # 移動平均
            moving_avg = [np.mean(self.scores[max(0, i-99):i+1]) for i in range(len(self.scores))]
            plt.plot(moving_avg, 'r-', label='100回合移動平均')
            plt.legend()
        plt.xlabel('回合')
        plt.ylabel('分數')
        plt.title('訓練分數進度')
        plt.grid(True)
        
        # Q表大小增長
        plt.subplot(1, 2, 2)
        q_table_sizes = list(range(1, len(self.q_table) + 1))
        plt.plot(q_table_sizes)
        plt.xlabel('狀態數量')
        plt.ylabel('Q表大小')
        plt.title('Q表增長')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('training_progress.png', dpi=300, bbox_inches='tight')
        plt.show()


if __name__ == "__main__":
    # 創建AI實例
    ai = SnakeAI(
        board_size=8,  # 較小的板子便於訓練
        learning_rate=0.1,
        discount_factor=0.95,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.01
    )
    
    # 訓練AI
    print("=== 貝爾曼方程Q-learning訓練 ===")
    scores = ai.train(num_episodes=2000, save_every=500, verbose=True)
    
    # 保存最終模型
    ai.save_model("snake_ai_final.pkl")
    
    # 繪製訓練進度
    ai.plot_training_progress()
    
    # 測試AI表現
    print("\n=== 測試AI表現 ===")
    test_scores = []
    for i in range(10):
        score, steps = ai.play_game(render=False)
        test_scores.append(score)
        print(f"測試 {i+1}: 分數={score}, 步數={steps}")
    
    print(f"\n平均測試分數: {np.mean(test_scores):.2f}")
    print(f"最高測試分數: {max(test_scores)}")