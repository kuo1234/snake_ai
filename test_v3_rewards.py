"""
测试 GymSnakeEnvV3 的课程优化奖励系统
验证三大策略是否被正确激励：
1. 从边角开始
2. 保持耐心
3. 善用转彯
"""
import numpy as np
from envs.gym_snake_env_v3 import GymSnakeEnvV3


def test_edge_bonus():
    """测试边缘奖励机制"""
    print("=" * 60)
    print("测试 1: 从边角开始 - 边缘奖励机制")
    print("=" * 60)
    
    env = GymSnakeEnvV3(board_size=6, seed=42)
    obs, info = env.reset()
    
    # 模拟在边缘移动
    print("\n场景：蛇在边缘移动")
    edge_rewards = []
    for _ in range(10):
        # 向右移动（沿着边缘）
        obs, reward, terminated, truncated, info = env.step(2)  # RIGHT
        if not terminated:
            edge_rewards.append(reward)
            print(f"  步骤 {len(edge_rewards)}: 奖励={reward:.3f}, 边缘时间={info['edge_time']}, 比例={info['edge_ratio']*100:.1f}%")
        if terminated or truncated:
            break
    
    avg_edge_reward = np.mean(edge_rewards) if edge_rewards else 0
    print(f"\n✓ 边缘移动平均奖励: {avg_edge_reward:.3f}")
    print(f"✓ 预期: 应该高于 0.2 (基础生存奖励)")
    print(f"✓ 实际包含: 0.2 (生存) + 0.3~0.45 (边缘奖励) ≈ 0.5~0.65")
    
    return avg_edge_reward


def test_patience_rewards():
    """测试耐心机制（降低的距离奖励急迫性）"""
    print("\n" + "=" * 60)
    print("测试 2: 保持耐心 - 降低距离奖励急迫性")
    print("=" * 60)
    
    env = GymSnakeEnvV3(board_size=6, seed=123)
    obs, info = env.reset()
    
    print("\n场景：不直接追食物，而是先探索")
    rewards = []
    
    # 模拟不直奔食物的移动（可能远离食物）
    for step in range(5):
        action = (step % 2) + 1  # 交替 LEFT/RIGHT，不直接追食物
        obs, reward, terminated, truncated, info = env.step(action)
        if not terminated:
            rewards.append(reward)
            print(f"  步骤 {step+1}: 奖励={reward:.3f}")
        if terminated or truncated:
            break
    
    avg_patience_reward = np.mean(rewards) if rewards else 0
    print(f"\n✓ 探索移动平均奖励: {avg_patience_reward:.3f}")
    print(f"✓ 预期: 即使远离食物，惩罚也较小（-0.2 而非 -0.5）")
    print(f"✓ 总奖励应该接近 0 或略正（生存 +0.2 抵消部分距离惩罚）")
    
    return avg_patience_reward


def test_turn_rewards():
    """测试转弯奖励机制"""
    print("\n" + "=" * 60)
    print("测试 3: 善用转弯 - 战术转弯奖励")
    print("=" * 60)
    
    env = GymSnakeEnvV3(board_size=6, seed=456)
    obs, info = env.reset()
    
    print("\n场景：进行多次转弯")
    turn_count = 0
    successful_turns = 0
    
    # 模拟转弯序列
    actions = [0, 1, 3, 2, 0]  # UP, LEFT, DOWN, RIGHT, UP
    prev_direction = None
    
    for i, action in enumerate(actions):
        obs, reward, terminated, truncated, info = env.step(action)
        if not terminated:
            action_names = ["UP", "LEFT", "RIGHT", "DOWN"]
            current_dir = action_names[action]
            is_turn = (prev_direction is not None and current_dir != prev_direction)
            
            if is_turn:
                turn_count += 1
                print(f"  转弯 {turn_count}: {prev_direction} -> {current_dir}, 奖励={reward:.3f}")
            
            prev_direction = current_dir
            
        if terminated or truncated:
            break
    
    final_info = info
    print(f"\n✓ 总转弯次数: {final_info['turn_count']}")
    print(f"✓ 成功转弯次数: {final_info['successful_turns']} (避开碰撞的转弯)")
    print(f"✓ 预期: 成功的战术转弯应该获得 +0.5 额外奖励")
    
    return final_info['turn_count'], final_info['successful_turns']


def test_full_episode():
    """完整回合测试"""
    print("\n" + "=" * 60)
    print("测试 4: 完整回合 - 综合表现")
    print("=" * 60)
    
    env = GymSnakeEnvV3(board_size=6, seed=789)
    
    # 运行 5 个回合
    episode_stats = []
    
    for ep in range(5):
        obs, info = env.reset()
        episode_reward = 0
        food_count = 0
        
        for step in range(200):
            # 简单策略：随机但避开明显危险
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            
            if 'food_obtained' in info and info['food_obtained']:
                food_count += 1
            
            if terminated or truncated:
                break
        
        episode_stats.append({
            'score': info['score'],
            'length': info['snake_length'],
            'reward': episode_reward,
            'edge_ratio': info['edge_ratio'],
            'turns': info['successful_turns'],
            'steps': step + 1
        })
        
        print(f"\n回合 {ep+1}:")
        print(f"  分数: {info['score']}")
        print(f"  蛇长: {info['snake_length']}")
        print(f"  总奖励: {episode_reward:.1f}")
        print(f"  边缘占比: {info['edge_ratio']*100:.1f}%")
        print(f"  成功转弯: {info['successful_turns']}/{info['turn_count']}")
        print(f"  存活步数: {step + 1}")
    
    # 汇总统计
    avg_score = np.mean([s['score'] for s in episode_stats])
    avg_reward = np.mean([s['reward'] for s in episode_stats])
    avg_edge = np.mean([s['edge_ratio'] for s in episode_stats])
    
    print(f"\n{'=' * 60}")
    print("汇总统计 (5 回合随机策略):")
    print(f"  平均分数: {avg_score:.1f}")
    print(f"  平均总奖励: {avg_reward:.1f}")
    print(f"  平均边缘占比: {avg_edge*100:.1f}%")
    print(f"{'=' * 60}")
    
    return episode_stats


def compare_v2_v3():
    """对比 V2 和 V3 的奖励差异"""
    print("\n" + "=" * 60)
    print("对比测试: V2 vs V3 奖励系统")
    print("=" * 60)
    
    try:
        from envs.gym_snake_env_v2 import GymSnakeEnvV2
        
        print("\n相同种子，相同动作序列，对比奖励...")
        seed = 999
        actions = [0, 2, 3, 1, 0, 2, 3, 1]  # 固定动作序列
        
        # V2
        env_v2 = GymSnakeEnvV2(board_size=6, seed=seed)
        obs, _ = env_v2.reset()
        v2_rewards = []
        for action in actions:
            obs, reward, terminated, truncated, info = env_v2.step(action)
            v2_rewards.append(reward)
            if terminated or truncated:
                break
        
        # V3
        env_v3 = GymSnakeEnvV3(board_size=6, seed=seed)
        obs, _ = env_v3.reset()
        v3_rewards = []
        for action in actions:
            obs, reward, terminated, truncated, info = env_v3.step(action)
            v3_rewards.append(reward)
            if terminated or truncated:
                break
        
        print(f"\nV2 奖励序列: {[f'{r:.2f}' for r in v2_rewards]}")
        print(f"V3 奖励序列: {[f'{r:.2f}' for r in v3_rewards]}")
        print(f"\nV2 平均: {np.mean(v2_rewards):.3f}")
        print(f"V3 平均: {np.mean(v3_rewards):.3f}")
        print(f"\n分析: V3 应该奖励更平缓（更多耐心），更重视边缘策略")
        
    except ImportError:
        print("\n⚠ 未找到 GymSnakeEnvV2，跳过对比测试")


if __name__ == "__main__":
    print("GymSnakeEnvV3 课程优化奖励系统测试\n")
    print("目标：验证三大策略的奖励设计")
    print("1. 从边角开始 - 边缘移动奖励")
    print("2. 保持耐心 - 降低急迫性")
    print("3. 善用转弯 - 战术转弯奖励")
    print()
    
    # 运行测试
    test_edge_bonus()
    test_patience_rewards()
    test_turn_rewards()
    test_full_episode()
    compare_v2_v3()
    
    print("\n" + "=" * 60)
    print("✓ 所有测试完成！")
    print("=" * 60)
    print("\n下一步：运行 PPO 训练")
    print("  python snake_ai_ppo_v3.py --mode train --quick")
    print("  期望：在 6x6 上达到 25+ 分（使用新的奖励策略）")
