"""
Q-table工作原理詳細說明
展示狀態表示和Q值查找過程
"""

import numpy as np
from snake_ai import SnakeAI
from snake_game import SnakeGame

def explain_state_representation():
    """詳細解釋狀態表示方法"""
    
    print("=== Q-table 狀態表示說明 ===")
    print()
    
    # 創建一個簡單的遊戲情況
    game = SnakeGame(silent_mode=True, board_size=6)
    game.reset()
    
    # 手動設置一個具體情況
    game.snake = [(2, 2), (2, 1), (2, 0)]  # 蛇頭在(2,2)，向右移動
    game.food = (1, 4)  # 食物在(1,4)
    game.direction = "RIGHT"
    
    ai = SnakeAI(board_size=6)
    
    print("遊戲情況:")
    print(f"蛇頭位置: {game.snake[0]}")
    print(f"食物位置: {game.food}")
    print(f"當前方向: {game.direction}")
    print()
    
    # 獲取狀態
    state = ai.get_state(game)
    
    print("狀態特徵解析:")
    print("───────────────────────────────────")
    
    # 危險檢測
    print("1. 危險檢測 (上下左右):")
    danger_features = state[0:4]
    directions = ["UP", "DOWN", "LEFT", "RIGHT"]
    for i, (direction, danger) in enumerate(zip(directions, danger_features)):
        status = "有危險" if danger else "安全"
        print(f"   {direction:5s}: {danger} ({status})")
    
    print()
    
    # 食物方向
    print("2. 食物相對方向:")
    food_features = state[4:8] 
    food_directions = ["UP", "DOWN", "LEFT", "RIGHT"]
    for i, (direction, has_food) in enumerate(zip(food_directions, food_features)):
        status = "有食物" if has_food else "無食物"
        print(f"   {direction:5s}: {has_food} ({status})")
    
    print()
    
    # 當前方向
    print("3. 當前移動方向:")
    dir_features = state[8:12]
    current_directions = ["UP", "DOWN", "LEFT", "RIGHT"] 
    for i, (direction, is_current) in enumerate(zip(current_directions, dir_features)):
        status = "當前方向" if is_current else ""
        print(f"   {direction:5s}: {is_current} {status}")
    
    print()
    print(f"完整狀態向量: {state}")
    
    return state

def demonstrate_different_scenarios():
    """演示不同遊戲情況下的狀態"""
    
    print("\n=== 不同場景的狀態比較 ===")
    
    scenarios = [
        {
            "name": "場景1: 食物在右上方",
            "snake": [(3, 3), (3, 2), (3, 1)],
            "food": (1, 5),
            "direction": "RIGHT"
        },
        {
            "name": "場景2: 食物在左下方",
            "snake": [(1, 4), (1, 3), (1, 2)],
            "food": (3, 2),
            "direction": "LEFT"
        },
        {
            "name": "場景3: 接近牆壁", 
            "snake": [(0, 3), (1, 3), (2, 3)],
            "food": (2, 5),
            "direction": "UP"
        }
    ]
    
    game = SnakeGame(silent_mode=True, board_size=6)
    ai = SnakeAI(board_size=6)
    
    for scenario in scenarios:
        print(f"\n{scenario['name']}:")
        print("─" * 40)
        
        # 設置場景
        game.snake = scenario['snake']
        game.food = scenario['food']
        game.direction = scenario['direction']
        
        state = ai.get_state(game)
        
        print(f"蛇頭: {game.snake[0]}, 食物: {game.food}, 方向: {game.direction}")
        print(f"狀態: {state}")
        
        # 解釋這個狀態
        head_row, head_col = game.snake[0]
        food_row, food_col = game.food
        
        print("分析:")
        if food_row < head_row:
            print("  - 食物在上方")
        elif food_row > head_row:
            print("  - 食物在下方")
            
        if food_col < head_col:
            print("  - 食物在左方")
        elif food_col > head_col:
            print("  - 食物在右方")

def show_qtable_lookup():
    """展示Q-table查找過程"""
    
    print("\n=== Q-table 查找過程 ===")
    
    # 載入訓練好的AI
    ai = SnakeAI(board_size=8)
    try:
        ai.load_model("snake_ai_standard.pkl")
        print("成功載入訓練好的模型")
    except:
        print("模型未找到，使用新的AI")
        # 手動添加一些Q值用於演示
        sample_state = (1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0)
        ai.q_table[sample_state] = np.array([10.5, -5.2, 8.3, 2.1])
    
    # 示例狀態
    example_states = [
        (1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0),  # 上方有危險，食物在上方，向右移動
        (0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0),  # 下方有危險，食物在下方，向上移動
        (0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0),  # 左方有危險，食物在左方，向下移動
    ]
    
    action_names = {0: "UP", 1: "LEFT", 2: "RIGHT", 3: "DOWN"}
    
    for i, state in enumerate(example_states):
        print(f"\n示例 {i+1}:")
        print(f"狀態: {state}")
        
        # 獲取Q值
        q_values = ai.get_q_values(state)
        print("Q值:")
        for action, q_val in enumerate(q_values):
            print(f"  {action_names[action]:5s}: {q_val:8.3f}")
        
        # AI會選擇的動作
        best_action = np.argmax(q_values)
        print(f"AI選擇: {action_names[best_action]} (Q值最高)")

def explain_key_insight():
    """解釋關鍵洞察"""
    
    print("\n" + "="*60)
    print("🔑 關鍵洞察：Q-table 不是按絕對位置索引的！")
    print("="*60)
    
    print("""
    ❌ 錯誤理解：
    "食物在位置A時查找A的Q-table"
    
    ✅ 正確理解：
    "根據蛇頭與食物的相對關係 + 危險情況 + 當前方向來查找Q值"
    
    為什麼這樣設計？
    
    1. 🎯 泛化能力：
       - 同樣的相對關係在不同位置都適用
       - 學到的策略可以應用到整個棋盤
    
    2. 📊 狀態空間縮減：
       - 8×8棋盤有64個位置，食物也有64個位置
       - 如果按絕對位置：64×64 = 4096種基本情況
       - 我們的方法：2^12 = 4096，但實際只有~256個有意義的狀態
    
    3. 🧠 學習效率：
       - 相似情況共享學習經驗
       - 更快收斂到最優策略
    
    舉例說明：
    ────────────────────────────────────────────
    情況A: 蛇頭(2,2)，食物(1,3) → 食物在右上方
    情況B: 蛇頭(5,1)，食物(4,2) → 食物在右上方  
    
    雖然絕對位置不同，但都對應同一個狀態特徵！
    AI學會處理"食物在右上方"的策略後，
    就能應用到所有類似的相對位置關係。
    ────────────────────────────────────────────
    """)

def interactive_state_explorer():
    """互動式狀態探索器"""
    
    print("\n=== 互動式狀態探索 ===")
    
    game = SnakeGame(silent_mode=True, board_size=6)
    ai = SnakeAI(board_size=6)
    
    # 嘗試載入模型
    try:
        ai.load_model("snake_ai_standard.pkl")
        has_model = True
    except:
        has_model = False
    
    print("您可以設置不同的遊戲情況來查看對應的狀態")
    print("輸入格式: 蛇頭行 蛇頭列 食物行 食物列 方向(UP/DOWN/LEFT/RIGHT)")
    print("示例: 2 2 4 5 RIGHT")
    print("輸入 'quit' 退出")
    
    action_names = {0: "UP", 1: "LEFT", 2: "RIGHT", 3: "DOWN"}
    
    while True:
        try:
            user_input = input("\n請輸入: ").strip()
            
            if user_input.lower() == 'quit':
                break
                
            parts = user_input.split()
            if len(parts) != 5:
                print("格式錯誤，請重新輸入")
                continue
                
            head_row, head_col = int(parts[0]), int(parts[1])
            food_row, food_col = int(parts[2]), int(parts[3])
            direction = parts[4].upper()
            
            # 設置遊戲狀態
            game.snake = [(head_row, head_col), (head_row, head_col-1)]
            game.food = (food_row, food_col)
            game.direction = direction
            
            # 獲取狀態
            state = ai.get_state(game)
            
            print(f"\n設置: 蛇頭({head_row},{head_col}), 食物({food_row},{food_col}), 方向{direction}")
            print(f"狀態向量: {state}")
            
            if has_model:
                q_values = ai.get_q_values(state)
                print("Q值:")
                for action, q_val in enumerate(q_values):
                    print(f"  {action_names[action]}: {q_val:8.3f}")
                best_action = np.argmax(q_values)
                print(f"AI推薦動作: {action_names[best_action]}")
            
        except ValueError:
            print("輸入格式錯誤，請重新輸入")
        except KeyboardInterrupt:
            break
    
    print("探索結束")

if __name__ == "__main__":
    # 執行所有說明
    explain_state_representation()
    demonstrate_different_scenarios() 
    show_qtable_lookup()
    explain_key_insight()
    
    # 可選的互動探索
    print("\n是否要進入互動式狀態探索？(y/n)")
    if input().lower().startswith('y'):
        interactive_state_explorer()