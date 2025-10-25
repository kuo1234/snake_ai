"""
貪吃蛇AI訓練腳本
使用貝爾曼方程的Q-learning算法
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from snake_ai import SnakeAI
from envs.snake_game import SnakeGame
import time

def quick_train():
    """快速訓練模式 - 較少回合用於測試"""
    print("=== 快速訓練模式 ===")
    ai = SnakeAI(
        board_size=6,      # 小板子
        learning_rate=0.15,
        discount_factor=0.9,
        epsilon=1.0,
        epsilon_decay=0.99,
        epsilon_min=0.05
    )
    
    # 訓練500回合
    ai.train(num_episodes=500, save_every=100)
    ai.save_model("snake_ai_quick.pkl")
    
    # 測試表現
    test_ai(ai, num_tests=5)
    
    return ai

def standard_train():
    """標準訓練模式"""
    print("=== 標準訓練模式 ===")
    ai = SnakeAI(
        board_size=8,
        learning_rate=0.1,
        discount_factor=0.95,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.01
    )
    
    # 訓練2000回合
    ai.train(num_episodes=2000, save_every=500)
    ai.save_model("snake_ai_standard.pkl")
    
    # 測試表現
    test_ai(ai, num_tests=10)
    
    return ai

def intensive_train():
    """深度訓練模式"""
    print("=== 深度訓練模式 ===")
    ai = SnakeAI(
        board_size=10,
        learning_rate=0.08,
        discount_factor=0.98,
        epsilon=1.0,
        epsilon_decay=0.9995,
        epsilon_min=0.005
    )
    
    # 訓練5000回合
    ai.train(num_episodes=5000, save_every=1000)
    ai.save_model("snake_ai_intensive.pkl")
    
    # 測試表現
    test_ai(ai, num_tests=20)
    
    return ai

def test_ai(ai, num_tests=10):
    """測試AI表現"""
    print(f"\n=== 測試AI ({num_tests}局) ===")
    
    scores = []
    steps_list = []
    
    for i in range(num_tests):
        score, steps = ai.play_game(render=False)
        scores.append(score)
        steps_list.append(steps)
        print(f"測試 {i+1:2d}: 分數={score:3d}, 步數={steps:4d}")
    
    print(f"\n統計結果:")
    print(f"平均分數: {np.mean(scores):.2f} ± {np.std(scores):.2f}")
    print(f"最高分數: {max(scores)}")
    print(f"最低分數: {min(scores)}")
    print(f"平均步數: {np.mean(steps_list):.1f}")

def compare_parameters():
    """比較不同參數設置的效果"""
    print("=== 參數比較實驗 ===")
    
    configs = [
        {"name": "高學習率", "lr": 0.2, "gamma": 0.9},
        {"name": "低學習率", "lr": 0.05, "gamma": 0.9},
        {"name": "高折扣", "lr": 0.1, "gamma": 0.99},
        {"name": "低折扣", "lr": 0.1, "gamma": 0.8},
    ]
    
    results = {}
    
    for config in configs:
        print(f"\n訓練 {config['name']} (lr={config['lr']}, γ={config['gamma']})")
        
        ai = SnakeAI(
            board_size=6,
            learning_rate=config['lr'],
            discount_factor=config['gamma'],
            epsilon=1.0,
            epsilon_decay=0.99,
            epsilon_min=0.05
        )
        
        # 短期訓練
        ai.train(num_episodes=300, save_every=0, verbose=False)
        
        # 測試
        scores = []
        for _ in range(10):
            score, _ = ai.play_game(render=False)
            scores.append(score)
        
        results[config['name']] = {
            'scores': scores,
            'avg': np.mean(scores),
            'std': np.std(scores)
        }
    
    # 顯示比較結果
    print("\n=== 參數比較結果 ===")
    for name, result in results.items():
        print(f"{name}: {result['avg']:.2f} ± {result['std']:.2f}")

def interactive_demo():
    """互動演示 - 載入模型並觀看AI遊戲"""
    print("=== 互動演示 ===")
    
    # 嘗試載入已訓練的模型
    ai = SnakeAI(board_size=8)
    
    models = ["snake_ai_final.pkl", "snake_ai_standard.pkl", "snake_ai_quick.pkl"]
    model_loaded = False
    
    for model_file in models:
        try:
            ai.load_model(model_file)
            model_loaded = True
            break
        except:
            continue
    
    if not model_loaded:
        print("未找到預訓練模型，進行快速訓練...")
        ai = quick_train()
    
    print("\n觀看AI遊戲 (按Ctrl+C停止)")
    
    try:
        game = SnakeGame(silent_mode=False, board_size=ai.board_size)
        
        while True:
            print("\n開始新遊戲...")
            score, steps = ai.play_game(game, render=True)
            print(f"遊戲結束! 分數: {score}, 步數: {steps}")
            
            # 短暫暫停
            time.sleep(2)
            
    except KeyboardInterrupt:
        print("\n演示結束")

def analyze_q_table(model_file):
    """分析Q表內容"""
    ai = SnakeAI()
    try:
        ai.load_model(model_file)
        
        print(f"\n=== Q表分析 ({model_file}) ===")
        print(f"總狀態數: {len(ai.q_table)}")
        
        # 找出最有價值的狀態-動作對
        all_q_values = []
        for state, q_vals in ai.q_table.items():
            for action, q_val in enumerate(q_vals):
                all_q_values.append((q_val, state, action))
        
        all_q_values.sort(reverse=True)
        
        print(f"最高Q值: {all_q_values[0][0]:.3f}")
        print(f"最低Q值: {all_q_values[-1][0]:.3f}")
        print(f"平均Q值: {np.mean([qv[0] for qv in all_q_values]):.3f}")
        
    except Exception as e:
        print(f"分析失敗: {e}")

def main():
    parser = argparse.ArgumentParser(description="貪吃蛇AI訓練")
    parser.add_argument("--mode", choices=['quick', 'standard', 'intensive', 'compare', 'demo', 'analyze'], 
                       default='standard', help="訓練模式")
    parser.add_argument("--model", type=str, help="模型文件 (用於分析)")
    
    args = parser.parse_args()
    
    if args.mode == 'quick':
        quick_train()
    elif args.mode == 'standard':
        standard_train()
    elif args.mode == 'intensive':
        intensive_train()
    elif args.mode == 'compare':
        compare_parameters()
    elif args.mode == 'demo':
        interactive_demo()
    elif args.mode == 'analyze':
        if args.model:
            analyze_q_table(args.model)
        else:
            print("請指定要分析的模型文件: --model <filename>")

if __name__ == "__main__":
    main()