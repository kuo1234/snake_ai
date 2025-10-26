"""
测试动态课程奖励系统
验证 Stage 1 (conservative) 和 Stage 2+ (aggressive) 的奖励差异
"""
import numpy as np
from envs.gym_snake_env_v3 import GymSnakeEnvV3


def test_conservative_stage():
    """测试保守模式 (Stage 1 - 6x6)"""
    print("=" * 70)
    print("测试 1: 保守模式 (Stage 1 - 6x6)")
    print("=" * 70)
    print("策略: 边缘优先，低急迫性，无饥饿惩罚\n")
    
    env = GymSnakeEnvV3(board_size=6, seed=42, curriculum_stage="conservative")
    obs, info = env.reset()
    
    # 模拟在边缘移动（不吃食物）
    print("场景 A: 在边缘移动（不吃食物）")
    rewards = []
    for i in range(10):
        action = 2  # RIGHT
        obs, reward, terminated, truncated, info = env.step(action)
        if not terminated:
            rewards.append(reward)
            if i < 3:
                print(f"  步骤 {i+1}: 奖励={reward:.3f}, 边缘时间={info['edge_time']}")
        if terminated or truncated:
            break
    
    avg_reward = np.mean(rewards) if rewards else 0
    print(f"\n边缘移动平均奖励: {avg_reward:.3f}")
    print(f"✓ 预期: 约 0.4-0.6 (0.2 生存 + 0.3 边缘奖励)")
    
    return avg_reward


def test_aggressive_stage():
    """测试积极模式 (Stage 2 - 8x8)"""
    print("\n" + "=" * 70)
    print("测试 2: 积极模式 (Stage 2 - 8x8)")
    print("=" * 70)
    print("策略: 食物优先，高急迫性，饥饿惩罚\n")
    
    env = GymSnakeEnvV3(board_size=8, seed=42, curriculum_stage="aggressive")
    obs, info = env.reset()
    
    # 模拟在边缘移动（不吃食物）
    print("场景 A: 在边缘移动（不吃食物）")
    rewards = []
    for i in range(10):
        action = 2  # RIGHT
        obs, reward, terminated, truncated, info = env.step(action)
        if not terminated:
            rewards.append(reward)
            if i < 3:
                print(f"  步骤 {i+1}: 奖励={reward:.3f}, 边缘时间={info['edge_time']}, 饥饿={info['steps_since_food']}")
        if terminated or truncated:
            break
    
    avg_reward = np.mean(rewards) if rewards else 0
    print(f"\n边缘移动平均奖励: {avg_reward:.3f}")
    print(f"✓ 预期: 约 -0.3 (0.1 生存 + 无边缘奖励 - 0.5 远离食物)")
    print(f"✓ 对比 Stage 1: 应该明显更低（移除边缘奖励）")
    
    # 测试饥饿惩罚
    print("\n场景 B: 测试饥饿惩罚")
    env2 = GymSnakeEnvV3(board_size=8, seed=123, curriculum_stage="aggressive")
    obs, info = env2.reset()
    
    hunger_threshold = 64  # 8x8
    # 模拟绕圈不吃食物
    for i in range(hunger_threshold + 5):
        action = (i % 4)  # 循环 UP, LEFT, RIGHT, DOWN
        obs, reward, terminated, truncated, info = env2.step(action)
        
        if i == hunger_threshold - 1:
            print(f"  步骤 {i+1} (阈值前): 饥饿={info['steps_since_food']}, 奖励={reward:.3f}")
        elif i == hunger_threshold:
            print(f"  步骤 {i+1} (阈值时): 饥饿={info['steps_since_food']}, 奖励={reward:.3f}")
            print(f"  ⚠️ 触发饥饿惩罚！应该看到 -5.0 额外惩罚")
        elif i == hunger_threshold + 1:
            print(f"  步骤 {i+1} (阈值后): 饥饿={info['steps_since_food']}, 奖励={reward:.3f}")
        
        if terminated or truncated:
            break
    
    return avg_reward


def test_distance_rewards():
    """对比两种模式的距离奖励"""
    print("\n" + "=" * 70)
    print("测试 3: 距离奖励对比")
    print("=" * 70)
    
    # Conservative (6x6)
    print("\n保守模式 (Stage 1):")
    env1 = GymSnakeEnvV3(board_size=6, seed=999, curriculum_stage="conservative")
    obs, info = env1.reset()
    
    # 故意远离食物
    distance_rewards_conservative = []
    for i in range(5):
        action = 1  # LEFT (可能远离食物)
        obs, reward, terminated, truncated, info = env1.step(action)
        if not terminated:
            distance_rewards_conservative.append(reward)
            if i < 3:
                print(f"  步骤 {i+1}: 奖励={reward:.3f}")
        if terminated or truncated:
            break
    
    avg_conservative = np.mean(distance_rewards_conservative) if distance_rewards_conservative else 0
    print(f"  平均: {avg_conservative:.3f} (预期: 接近 0，-0.2 远离惩罚较小)")
    
    # Aggressive (8x8)
    print("\n积极模式 (Stage 2):")
    env2 = GymSnakeEnvV3(board_size=8, seed=999, curriculum_stage="aggressive")
    obs, info = env2.reset()
    
    # 故意远离食物
    distance_rewards_aggressive = []
    for i in range(5):
        action = 1  # LEFT (可能远离食物)
        obs, reward, terminated, truncated, info = env2.step(action)
        if not terminated:
            distance_rewards_aggressive.append(reward)
            if i < 3:
                print(f"  步骤 {i+1}: 奖励={reward:.3f}")
        if terminated or truncated:
            break
    
    avg_aggressive = np.mean(distance_rewards_aggressive) if distance_rewards_aggressive else 0
    print(f"  平均: {avg_aggressive:.3f} (预期: 约 -0.4，-0.5 远离惩罚较大)")
    
    print(f"\n对比: 保守 {avg_conservative:.3f} vs 积极 {avg_aggressive:.3f}")
    print(f"✓ 积极模式应该明显更负（惩罚远离食物）")


def test_summary():
    """总结对比"""
    print("\n" + "=" * 70)
    print("测试 4: 总结对比")
    print("=" * 70)
    
    print("\n保守模式 (Stage 1 - 6x6):")
    print("  ✓ 边缘奖励: +0.3 * multiplier")
    print("  ✓ 靠近食物: +0.3 (低急迫性)")
    print("  ✓ 远离食物: -0.2 (温和惩罚)")
    print("  ✓ 生存奖励: +0.2")
    print("  ✓ 饥饿惩罚: 无")
    print("  → 策略: 边缘为主，慢慢靠近食物")
    
    print("\n积极模式 (Stage 2+ - 8x8+):")
    print("  ✗ 边缘奖励: 移除")
    print("  ✓ 靠近食物: +1.0 (V2 风格高急迫性)")
    print("  ✓ 远离食物: -0.5 (V2 风格强惩罚)")
    print("  ✓ 生存奖励: +0.1 (降低)")
    print("  ✓ 饥饿惩罚: -5.0 (64 步内不吃食物)")
    print("  → 策略: 主动追食物，不能拖延")


if __name__ == "__main__":
    print("PPO V3 动态课程奖励系统测试\n")
    print("目标：验证 Stage 1 和 Stage 2+ 的奖励差异")
    print("Stage 1: 保守策略（边缘优先）")
    print("Stage 2+: 积极策略（食物优先 + 饥饿惩罚）\n")
    
    # 运行测试
    test_conservative_stage()
    test_aggressive_stage()
    test_distance_rewards()
    test_summary()
    
    print("\n" + "=" * 70)
    print("✓ 所有测试完成！")
    print("=" * 70)
    print("\n下一步：运行训练")
    print("  python snake_ai_ppo_v3.py --mode train")
    print("  期望: Stage 1 学会边缘策略，Stage 2 切换到积极追食")
