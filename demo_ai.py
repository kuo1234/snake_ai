"""
貪吃蛇AI可視化演示
觀看訓練好的AI如何遊戲
"""

import sys
import time
import pygame
from snake_ai import SnakeAI
from snake_game import SnakeGame

def watch_ai_play():
    """觀看AI遊戲並顯示決策過程"""
    
    # 載入訓練好的AI
    ai = SnakeAI(board_size=8)
    
    # 嘗試載入最佳模型
    models_to_try = [
        "snake_ai_standard.pkl",
        "snake_ai_quick.pkl", 
        "snake_ai_final.pkl"
    ]
    
    model_loaded = False
    for model_file in models_to_try:
        try:
            ai.load_model(model_file)
            print(f"成功載入模型: {model_file}")
            model_loaded = True
            break
        except:
            continue
    
    if not model_loaded:
        print("警告: 未找到預訓練模型，使用隨機AI")
    
    # 創建遊戲實例
    game = SnakeGame(silent_mode=False, board_size=ai.board_size)
    
    action_names = {0: "UP", 1: "LEFT", 2: "RIGHT", 3: "DOWN"}
    
    print("=== AI貪吃蛇演示 ===")
    print("觀看AI如何使用貝爾曼方程學到的策略")
    print("按ESC鍵或關閉窗口結束演示")
    
    try:
        game_count = 1
        
        while True:
            print(f"\n--- 遊戲 {game_count} 開始 ---")
            
            # 重置遊戲
            game.reset()
            state = ai.get_state(game)
            steps = 0
            max_steps = ai.board_size * ai.board_size * 3
            
            # 遊戲循環
            while steps < max_steps:
                # AI選擇動作
                action = ai.choose_action(state, training=False)
                q_values = ai.get_q_values(state)
                
                # 顯示AI的思考過程
                print(f"步驟 {steps+1:3d}: 動作={action_names[action]:5s} "
                      f"Q值=[{q_values[0]:6.2f}, {q_values[1]:6.2f}, {q_values[2]:6.2f}, {q_values[3]:6.2f}] "
                      f"分數={game.score:3d}")
                
                # 執行動作
                done, info = game.step(action)
                
                # 渲染遊戲
                if hasattr(game, 'render') and game.screen:
                    game.render()
                
                # 檢查pygame事件
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        return
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE:
                            pygame.quit()
                            return
                        elif event.key == pygame.K_SPACE:
                            # 空格鍵暫停/繼續
                            input("按Enter繼續...")
                
                if done:
                    print(f"遊戲結束! 最終分數: {game.score}, 總步數: {steps+1}")
                    
                    # 等待一下再開始下一局
                    time.sleep(2)
                    break
                
                # 更新狀態
                state = ai.get_state(game)
                steps += 1
                
                # 控制遊戲速度
                time.sleep(0.2)
            
            game_count += 1
            
            if game_count > 5:  # 最多觀看5局
                print("演示結束")
                break
                
    except KeyboardInterrupt:
        print("\n演示被用戶中斷")
    
    finally:
        if pygame.get_init():
            pygame.quit()

def compare_before_after_training():
    """比較訓練前後的AI表現"""
    
    print("=== 訓練前後比較 ===")
    
    # 未訓練的AI
    untrained_ai = SnakeAI(board_size=8)
    
    # 訓練好的AI
    trained_ai = SnakeAI(board_size=8)
    try:
        trained_ai.load_model("snake_ai_standard.pkl")
        print("載入訓練好的AI模型")
    except:
        print("警告: 未找到訓練好的模型")
        return
    
    # 測試兩個AI
    def test_ai_performance(ai, name, num_games=5):
        scores = []
        print(f"\n測試 {name}:")
        
        for i in range(num_games):
            score, steps = ai.play_game(render=False)
            scores.append(score)
            print(f"  遊戲 {i+1}: 分數={score}, 步數={steps}")
        
        avg_score = sum(scores) / len(scores)
        print(f"  平均分數: {avg_score:.2f}")
        return scores
    
    untrained_scores = test_ai_performance(untrained_ai, "未訓練AI")
    trained_scores = test_ai_performance(trained_ai, "訓練後AI")
    
    print(f"\n=== 比較結果 ===")
    print(f"未訓練AI平均分數: {sum(untrained_scores)/len(untrained_scores):.2f}")
    print(f"訓練後AI平均分數: {sum(trained_scores)/len(trained_scores):.2f}")
    print(f"改進倍數: {(sum(trained_scores)/len(trained_scores)) / max(1, sum(untrained_scores)/len(untrained_scores)):.2f}x")

def analyze_ai_strategy():
    """分析AI學到的策略"""
    
    ai = SnakeAI(board_size=8)
    try:
        ai.load_model("snake_ai_standard.pkl")
    except:
        print("需要先訓練AI模型")
        return
    
    print("=== AI策略分析 ===")
    print(f"Q表大小: {len(ai.q_table)} 個狀態")
    
    # 分析Q值分布
    all_q_values = []
    for state_q_values in ai.q_table.values():
        all_q_values.extend(state_q_values)
    
    import numpy as np
    print(f"Q值統計:")
    print(f"  最大值: {max(all_q_values):.2f}")
    print(f"  最小值: {min(all_q_values):.2f}")
    print(f"  平均值: {np.mean(all_q_values):.2f}")
    print(f"  標準差: {np.std(all_q_values):.2f}")
    
    # 找出最重要的狀態
    print(f"\n最有價值的狀態-動作對:")
    state_action_values = []
    action_names = {0: "UP", 1: "LEFT", 2: "RIGHT", 3: "DOWN"}
    
    for state, q_values in ai.q_table.items():
        for action, q_value in enumerate(q_values):
            state_action_values.append((q_value, state, action))
    
    state_action_values.sort(reverse=True)
    
    for i, (q_value, state, action) in enumerate(state_action_values[:5]):
        print(f"  {i+1}. Q值={q_value:.2f}, 動作={action_names[action]}, 狀態={state}")

def main():
    """主函數"""
    
    print("貪吃蛇AI演示程序")
    print("請選擇演示模式:")
    print("1. 觀看AI遊戲 (圖形界面)")
    print("2. 訓練前後比較")
    print("3. AI策略分析")
    
    try:
        choice = input("請輸入選擇 (1-3): ").strip()
        
        if choice == '1':
            watch_ai_play()
        elif choice == '2':
            compare_before_after_training()
        elif choice == '3':
            analyze_ai_strategy()
        else:
            print("無效選擇")
            
    except KeyboardInterrupt:
        print("\n程序被用戶中斷")

if __name__ == "__main__":
    main()