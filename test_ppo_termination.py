"""快速测试 PPO V3 模型，验证游戏不会提前结束"""
import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'

from stable_baselines3 import PPO
from envs.gym_snake_env_v2 import GymSnakeEnvV2

print("=== 测试 PPO V3 游戏是否会提前结束 ===\n")

# 加载模型
model_path = "models/ppo_snake_v3_curriculum/Stage1_Novice/model.zip"
if not os.path.exists(model_path):
    print(f"❌ 模型不存在: {model_path}")
    exit(1)

model = PPO.load(model_path)
print(f"✓ 载入模型: {model_path}\n")

# 创建环境
board_size = 6
env = GymSnakeEnvV2(board_size=board_size, render_mode=None)

print(f"棋盘大小: {board_size}x{board_size}")
print(f"最大步数: {env.max_episode_length}\n")

# 运行3局游戏
action_names = {0: "UP", 1: "LEFT", 2: "RIGHT", 3: "DOWN"}

for game_num in range(3):
    print(f"--- 游戏 {game_num + 1} ---")
    obs, info = env.reset()
    done = False
    steps = 0
    
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        steps += 1
    
    # 检查结束原因
    final_score = info.get('score', 0)
    final_length = info.get('snake_length', 0)
    
    print(f"  步数: {steps}")
    print(f"  分数: {final_score}")
    print(f"  蛇长: {final_length}")
    
    if truncated and not terminated:
        print(f"  ⚠️  游戏被步数限制强制结束（truncated）")
    elif terminated:
        if final_length >= env.game.grid_size:
            print(f"  🏆 游戏正常结束：获胜！")
        else:
            print(f"  ✓ 游戏正常结束：碰撞")
    
    print()

print("=== 测试完成 ===")
print("\n如果看到 '⚠️ 游戏被步数限制强制结束'，说明还有问题")
print("如果都是 '✓ 游戏正常结束'，说明问题已修复！")
