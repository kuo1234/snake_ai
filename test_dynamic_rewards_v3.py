"""测试动态奖励系统 (V3 Stage 1-4)"""
import sys
sys.path.append('.')

from envs.gym_snake_env_v3 import GymSnakeEnvV3
import numpy as np

print("="*70)
print("测试 V3 动态奖励系统 (Stage 1-4)")
print("="*70)

# 测试每个 Stage 的奖励系数
stages_config = [
    (1, 6, "保守 (Conservative)"),
    (2, 8, "积极 (Aggressive)"),
    (3, 10, "进阶 (Advanced)"),
    (4, 12, "大师 (Master)")
]

for stage, board_size, description in stages_config:
    print(f"\n{'='*70}")
    print(f"Stage {stage}: {board_size}x{board_size} - {description}")
    print(f"{'='*70}")
    
    # 创建环境
    env = GymSnakeEnvV3(board_size=board_size, render_mode=None, stage=stage)
    
    print(f"✓ 环境创建成功")
    print(f"  - Stage: {env.stage}")
    print(f"  - Board Size: {env.board_size}")
    print(f"  - Hunger Limit: {env.hunger_limit}")
    
    # 测试一个回合
    obs, info = env.reset()
    done = False
    step_count = 0
    total_reward = 0
    
    while not done and step_count < 20:
        # 随机动作
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        step_count += 1
        total_reward += reward
    
    print(f"  - 测试回合: {step_count} 步")
    print(f"  - 总奖励: {total_reward:.2f}")
    print(f"  - 饥饿计时器: {env.hunger_timer}")
    
    env.close()

print(f"\n{'='*70}")
print("✅ 所有 Stage 测试完成！")
print(f"{'='*70}")

# 测试动态切换
print(f"\n{'='*70}")
print("测试动态 Stage 切换")
print(f"{'='*70}")

env = GymSnakeEnvV3(board_size=6, render_mode=None, stage=1)
print(f"初始 Stage: {env.stage}")

# 切换到 Stage 2
obs, info = env.reset(stage=2)
print(f"切换后 Stage: {env.stage}")
print(f"饥饿计时器已重置: {env.hunger_timer}")

# 切换到 Stage 3
obs, info = env.reset(stage=3)
print(f"再次切换 Stage: {env.stage}")

env.close()
print("\n✅ 动态切换测试完成！")
