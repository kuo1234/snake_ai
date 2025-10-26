"""测试游戏是否会因为步数限制而提前结束"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from envs.gym_snake_env_v2 import GymSnakeEnvV2

print("=== 测试游戏步数限制 ===\n")

# 测试不同棋盘大小
for board_size in [6, 8, 10]:
    print(f"\n棋盘大小: {board_size}x{board_size}")
    env = GymSnakeEnvV2(board_size=board_size, render_mode=None)
    
    max_steps = env.max_episode_length
    grid_size = board_size * board_size
    
    print(f"  棋盘格子数: {grid_size}")
    print(f"  最大步数限制: {max_steps}")
    print(f"  步数/格子比: {max_steps / grid_size:.1f}x")
    
    # 理论上，在最优情况下，蛇需要访问所有格子
    # 所以步数限制应该远大于格子数
    if max_steps < grid_size * 5:
        print(f"  ⚠️  警告: 步数限制可能太小（建议至少 {grid_size * 5}）")
    else:
        print(f"  ✓ 步数限制合理")

print("\n" + "="*60)
print("修复前: max_episode_length = board_size² × 3")
print("修复后: max_episode_length = board_size² × 10")
print("="*60)

print("\n示例对比 (6x6 棋盘):")
print("  修复前: 36 × 3 = 108 步 ❌ (从日志看游戏刚好在108步结束)")
print("  修复后: 36 × 10 = 360 步 ✓ (足够蛇完成游戏)")
