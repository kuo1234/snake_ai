"""
快速测试 PPO V3 在 Stage 1 (6x6) 的训练效果
使用更短的训练时间来快速验证奖励优化
"""
import os
import sys
import time
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

from envs.gym_snake_env_v3 import GymSnakeEnvV3


def quick_train():
    """快速训练测试 (50k steps)"""
    print("=" * 60)
    print("PPO V3 快速训练测试 - Stage 1 (6x6)")
    print("=" * 60)
    print("\n目标：验证课程优化奖励在小地图上的效果")
    print("训练步数：50,000 (约 5 分钟)")
    print("期望分数：> 15 分（18/35 是当前瓶颈，目标突破 20+）\n")
    
    # 创建环境
    def make_env():
        env = GymSnakeEnvV3(board_size=6, render_mode=None, seed=42)
        env = Monitor(env)
        return env
    
    env = DummyVecEnv([make_env])
    
    # 创建模型（较小网络以加快训练）
    model = PPO(
        "MlpPolicy",
        env,
        policy_kwargs=dict(
            net_arch=dict(pi=[128, 128], vf=[128, 128])
        ),
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        verbose=1,
        tensorboard_log="./logs/ppo_v3_quick_test/"
    )
    
    print("\n开始训练...")
    start_time = time.time()
    model.learn(total_timesteps=50000)
    train_time = time.time() - start_time
    
    print(f"\n✓ 训练完成！用时: {train_time/60:.1f} 分钟")
    
    # 保存模型
    os.makedirs("models/quick_test", exist_ok=True)
    model_path = "models/quick_test/ppo_v3_stage1_quick"
    model.save(model_path)
    print(f"✓ 模型已保存: {model_path}")
    
    # 评估
    print("\n" + "=" * 60)
    print("评估训练后的模型...")
    print("=" * 60)
    
    eval_env = GymSnakeEnvV3(board_size=6, render_mode=None)
    scores = []
    lengths = []
    edge_ratios = []
    successful_turns_list = []
    
    for ep in range(20):
        obs, info = eval_env.reset()
        done = False
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = eval_env.step(action)
            done = terminated or truncated
        
        scores.append(info['score'])
        lengths.append(info['snake_length'])
        edge_ratios.append(info['edge_ratio'])
        successful_turns_list.append(info['successful_turns'])
    
    # 统计结果
    avg_score = np.mean(scores)
    max_score = np.max(scores)
    min_score = np.min(scores)
    std_score = np.std(scores)
    avg_length = np.mean(lengths)
    avg_edge = np.mean(edge_ratios)
    avg_turns = np.mean(successful_turns_list)
    
    print(f"\n评估结果 (20 回合):")
    print(f"  平均分数: {avg_score:.1f} ± {std_score:.1f}")
    print(f"  最高分数: {max_score}")
    print(f"  最低分数: {min_score}")
    print(f"  平均蛇长: {avg_length:.1f}")
    print(f"  边缘占比: {avg_edge*100:.1f}%")
    print(f"  平均成功转弯: {avg_turns:.1f}")
    
    print("\n" + "=" * 60)
    print("分析：")
    
    if avg_score < 15:
        print("⚠ 分数偏低 (<15)，可能需要：")
        print("  - 更长的训练时间")
        print("  - 调整边缘奖励系数")
        print("  - 增加探索（ent_coef）")
    elif avg_score < 20:
        print("✓ 分数良好 (15-20)，接近目标！")
        print("  - 继续训练可能突破 20+")
        print("  - 考虑微调超参数")
    elif avg_score < 25:
        print("✓✓ 分数优秀 (20-25)，已突破瓶颈！")
        print("  - V3 奖励优化有效")
        print("  - 可以进入完整训练")
    else:
        print("✓✓✓ 分数卓越 (25+)，超出预期！")
        print("  - 达到 Stage 1 毕业标准")
        print("  - 准备进入 Stage 2")
    
    print("=" * 60)
    
    return avg_score, model


def compare_untrained():
    """对比未训练模型的表现"""
    print("\n" + "=" * 60)
    print("对照组：未训练模型评估")
    print("=" * 60)
    
    env = GymSnakeEnvV3(board_size=6, render_mode=None, seed=999)
    
    # 创建未训练模型
    untrained_model = PPO(
        "MlpPolicy",
        DummyVecEnv([lambda: env]),
        policy_kwargs=dict(
            net_arch=dict(pi=[128, 128], vf=[128, 128])
        ),
        learning_rate=3e-4,
        verbose=0
    )
    
    scores = []
    for _ in range(10):
        obs, info = env.reset()
        done = False
        
        while not done:
            action, _ = untrained_model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        
        scores.append(info['score'])
    
    avg_untrained = np.mean(scores)
    print(f"\n未训练模型平均分数: {avg_untrained:.1f}")
    print(f"（对照基线，应该接近 0）")
    
    return avg_untrained


if __name__ == "__main__":
    print("PPO V3 快速训练测试\n")
    print("测试目的：")
    print("1. 验证 V3 环境和奖励系统可以正常训练")
    print("2. 初步评估能否突破 18/35 的瓶颈")
    print("3. 为完整训练提供参考\n")
    
    # 运行对照组
    baseline_score = compare_untrained()
    
    # 运行快速训练
    trained_score, model = quick_train()
    
    # 对比
    improvement = trained_score - baseline_score
    
    print(f"\n" + "=" * 60)
    print("最终总结:")
    print("=" * 60)
    print(f"未训练基线: {baseline_score:.1f} 分")
    print(f"训练后得分: {trained_score:.1f} 分")
    print(f"提升幅度:   +{improvement:.1f} 分")
    
    if improvement > 15:
        print("\n✓✓ 训练效果显著！V3 奖励系统工作良好。")
        print("建议：运行完整训练 (50万步)")
        print("  python snake_ai_ppo_v3.py --mode train")
    elif improvement > 10:
        print("\n✓ 训练有效，但有改进空间。")
        print("建议：尝试调整超参数或增加训练步数")
    else:
        print("\n⚠ 提升有限，需要诊断问题。")
        print("检查：奖励函数、观察空间、网络架构")
    
    print("=" * 60)
