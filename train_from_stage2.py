"""
快速启动脚本：从 Stage 2 开始训练
前提：Stage 1 已训练完成
"""

import subprocess
import sys
import os

def check_stage1_model():
    """检查 Stage 1 模型是否存在"""
    stage1_paths = [
        "models/ppo_snake_v3_curriculum/Stage1_Novice/best_model/best_model.zip",
        "models/ppo_snake_v3_curriculum/Stage1_Novice/model.zip"
    ]
    
    for path in stage1_paths:
        if os.path.exists(path):
            print(f"✓ 找到 Stage 1 模型: {path}")
            return True
    
    print("✗ 找不到 Stage 1 模型")
    print("可能的路径:")
    for path in stage1_paths:
        print(f"  - {path}")
    return False

def main():
    print("="*70)
    print("🚀 PPO V3 - 从 Stage 2 开始训练")
    print("="*70)
    
    # 检查 Stage 1 模型
    if not check_stage1_model():
        print("\n⚠️  警告: Stage 1 模型不存在")
        print("请先训练 Stage 1:")
        print("  python snake_ai_ppo_v3.py --mode train --start-stage 0")
        print("\n或者确认模型路径正确")
        return
    
    print("\n✓ Stage 1 模型已准备好")
    print("\n开始从 Stage 2 训练...")
    print("="*70)
    print()
    
    # 启动训练（从 Stage 2 开始，即 start-stage 1）
    cmd = [
        sys.executable,
        "snake_ai_ppo_v3.py",
        "--mode", "train",
        "--start-stage", "1",  # Stage 2 (索引从 0 开始)
        "--device", "auto"
    ]
    
    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\n\n训练被用户中断")
    except subprocess.CalledProcessError as e:
        print(f"\n\n训练失败: {e}")

if __name__ == "__main__":
    main()
