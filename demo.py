"""
贪吃蛇游戏演示脚本
"""

import sys
import os

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from envs.snake_game import SnakeGame

def demo_game():
    """演示游戏基本功能"""
    print("=== 贪吃蛇游戏演示 ===")
    print("游戏特点：")
    print("1. 使用方向键控制蛇的移动")
    print("2. 吃红色食物得分")
    print("3. 撞墙或撞到自己游戏结束")
    print("4. 点击START开始游戏")
    print("5. 游戏结束后点击RETRY重新开始")
    print("\n启动游戏...")
    
    # 创建游戏实例
    game = SnakeGame(silent_mode=False, board_size=12)
    
    print("游戏窗口已打开，请在游戏窗口中游玩！")
    print("按Ctrl+C停止演示脚本")
    
    try:
        # 这里可以添加自动游戏逻辑
        import time
        while True:
            time.sleep(1)  # 保持脚本运行
    except KeyboardInterrupt:
        print("\n演示结束")

def silent_demo():
    """静默模式演示（无图形界面）"""
    print("=== 静默模式演示 ===")
    game = SnakeGame(silent_mode=True, board_size=8)
    
    actions = [3, 2, 2, 0, 0, 1, 1, 3]  # 示例动作序列
    
    for i, action in enumerate(actions):
        print(f"步骤 {i+1}: 执行动作 {action}")
        done, info = game.step(action)
        print(f"  蛇长度: {info['snake_size']}")
        print(f"  蛇头位置: {info['snake_head_pos']}")
        print(f"  食物位置: {info['food_pos']}")
        print(f"  是否吃到食物: {info['food_obtained']}")
        print(f"  游戏结束: {done}")
        print(f"  当前分数: {game.score}")
        print("-" * 30)
        
        if done:
            print("游戏结束！")
            break

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="贪吃蛇游戏演示")
    parser.add_argument("--silent", action="store_true", help="运行静默模式演示")
    
    args = parser.parse_args()
    
    if args.silent:
        silent_demo()
    else:
        demo_game()