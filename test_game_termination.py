"""测试游戏结束逻辑是否正确"""
from envs.snake_game import SnakeGame
import random

print("=== 测试游戏结束逻辑 ===\n")

# 测试 1: 随机游戏直到结束
print("测试 1: 随机游戏（应该在碰撞时结束）")
game = SnakeGame(board_size=6, silent_mode=True)
steps = 0
max_steps = 200

while steps < max_steps:
    action = random.randint(0, 3)
    done, info = game.step(action)
    steps += 1
    
    if done:
        is_win = len(game.snake) >= game.grid_size
        print(f"✓ 游戏在第 {steps} 步正确结束")
        print(f"  分数: {game.score}")
        print(f"  蛇长: {len(game.snake)}")
        print(f"  棋盘大小: {game.grid_size}")
        print(f"  结束原因: {'🏆 获胜（填满棋盘）' if is_win else '💥 碰撞'}")
        break
else:
    print(f"✗ 游戏在 {max_steps} 步后仍未结束（异常）")

# 测试 2: 模拟接近获胜的情况
print("\n测试 2: 检查获胜条件")
game2 = SnakeGame(board_size=4, silent_mode=True)  # 小棋盘更容易测试
print(f"  棋盘大小: 4x4 (grid_size={game2.grid_size})")
print(f"  初始蛇长: {len(game2.snake)}")
print(f"  需要达到蛇长 >= {game2.grid_size} 才算获胜")

# 手动测试几步
for i in range(5):
    action = 2  # 向右
    done, info = game2.step(action)
    if done:
        is_win = len(game2.snake) >= game2.grid_size
        print(f"\n✓ 游戏在第 {i+1} 步结束")
        print(f"  蛇长: {len(game2.snake)}")
        print(f"  结束原因: {'🏆 获胜' if is_win else '💥 碰撞'}")
        break

print("\n=== 测试完成 ===")
