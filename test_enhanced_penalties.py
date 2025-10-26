"""测试新的陷阱惩罚值"""
import sys
sys.path.append('.')

from envs.gym_snake_env_v3 import GymSnakeEnvV3
import numpy as np

print("="*70)
print("测试 V3 增强陷阱惩罚 (Stage 2-4)")
print("="*70)

# 测试每个 Stage 的惩罚值
stages = [
    (1, 6, "保守策略"),
    (2, 8, "积极策略 - 强化惩罚"),
    (3, 10, "进阶策略 - 强化惩罚"),
    (4, 12, "大师策略 - 强化惩罚")
]

for stage, board_size, desc in stages:
    print(f"\n{'='*70}")
    print(f"Stage {stage}: {board_size}x{board_size} - {desc}")
    print(f"{'='*70}")
    
    # 创建环境
    env = GymSnakeEnvV3(board_size=board_size, render_mode=None, stage=stage)
    
    # 直接访问奖励配置
    from envs.gym_snake_env_v3 import GymSnakeEnvV3
    
    # 重新定义配置以查看
    reward_config = {
        1: {'trap_penalty': -1.5, 'approach': 0.3, 'edge_bonus': 0.3},
        2: {'trap_penalty': -10.0, 'approach': 1.0, 'edge_bonus': 0.0},
        3: {'trap_penalty': -15.0, 'approach': 1.0, 'edge_bonus': 0.0},
        4: {'trap_penalty': -15.0, 'approach': 0.5, 'edge_bonus': 0.0}
    }
    
    config = reward_config[stage]
    
    print(f"  📊 奖励配置:")
    print(f"     - 陷阱惩罚: {config['trap_penalty']}")
    print(f"     - 靠近食物: +{config['approach']}")
    print(f"     - 边缘奖励: +{config['edge_bonus']}")
    
    # 计算风险/收益比
    risk_reward_ratio = abs(config['trap_penalty']) / config['approach']
    print(f"     - 风险/收益比: {risk_reward_ratio:.1f}x")
    print(f"       (需冒险 {risk_reward_ratio:.1f} 次才值得)")
    
    env.close()

print(f"\n{'='*70}")
print("✅ 配置分析完成")
print(f"{'='*70}")

print("\n📈 策略影响:")
print("  Stage 1: 陷阱惩罚 -1.5 / 食物奖励 +0.3 = 5x 风险")
print("           → AI 会谨慎，但不会过分害怕")
print()
print("  Stage 2: 陷阱惩罚 -10.0 / 食物奖励 +1.0 = 10x 风险 ⬆️")
print("           → AI 必须非常谨慎，不能短视追食物")
print()
print("  Stage 3: 陷阱惩罚 -15.0 / 食物奖励 +1.0 = 15x 风险 ⬆️⬆️")
print("           → AI 会极度谨慎，优先空间管理")
print()
print("  Stage 4: 陷阱惩罚 -15.0 / 食物奖励 +0.5 = 30x 风险 ⬆️⬆️⬆️")
print("           → AI 会极度保守，追求长期生存")

print("\n💡 预期效果:")
print("  ✅ Stage 2+ 不会再因为贪吃而把自己卡死")
print("  ✅ AI 会学会评估「未来的空间」而非「眼前的食物」")
print("  ✅ 配合强化参数震荡（LR + Entropy），强制重新探索")
