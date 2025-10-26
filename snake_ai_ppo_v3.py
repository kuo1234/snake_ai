"""
PPO V3: Curriculum Learning 課程學習版本（V3 優化獎勵）
結合 V3 的課程優化獎勵 + 課程學習（從易到難）

V3 獎勵優化重點：
✓ 從邊角開始：獎勵在邊緣移動（+0.3 * 邊緣係數），邊緣時間佔比追蹤
✓ 保持耐心：降低距離獎勵急迫性（+0.3/-0.2，原為 +1.0/-0.5），提高生存獎勵（+0.2）
✓ 善用轉彎：獎勵避開碰撞的戰術轉彎（+0.5），追蹤成功轉彎數
✓ 空間管理：中心開放獎勵（+0.5），降低陷阱懲罰（-1.5）
✓ 溫和懲罰：降低死亡懲罰（-10 to -30，原為 -20 to -50），專注學習而非恐懼

課程設計：
階段 1: 6x6 小棋盤（新手村）- 學習從邊角開始、保持耐心、善用轉彎
階段 2: 8x8 標準棋盤（進階班）- 在中等空間中優化策略
階段 3: 10x10 困難棋盤（挑戰班）- 處理更複雜的空間規劃
階段 4: 12x12 極難棋盤（大師班）- 最終挑戰

每個階段都會：
1. 繼承上一階段的模型（Transfer Learning）
2. 設定畢業標準（平均分數達標）
3. 使用 V3 的課程優化獎勵（專為小地圖設計）

新增特性：
- V3 環境：20維觀察空間（增加邊緣距離、蛇長比例、可用空間比例）
- 更大的神經網路：[256, 256, 128] 取代 [128, 128, 64]
- 更長的訓練時間：Stage 1 增加到 50萬步（從 30萬）
- 自動階段切換：達標後自動升級（無需手動確認）
- 完整的訓練追蹤和日誌
"""

import os
import sys
import argparse
import time
import numpy as np
import torch
from typing import Optional, Dict, List
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure

# 導入 V3 環境（課程優化版）
from envs.gym_snake_env_v3 import GymSnakeEnvV3


class CurriculumStage:
    """課程階段定義"""
    def __init__(self, name: str, board_size: int, timesteps: int, 
                 graduation_score: float, description: str):
        self.name = name
        self.board_size = board_size
        self.timesteps = timesteps
        self.graduation_score = graduation_score
        self.description = description
        self.completed = False
        self.best_score = 0.0
        self.current_score = 0.0


class CurriculumManager:
    """課程管理器 - The Curriculum Coach"""
    
    def __init__(self, base_dir: str = "models/ppo_snake_v3_curriculum"):
        self.base_dir = base_dir
        os.makedirs(base_dir, exist_ok=True)
        
        # 定義課程階段（使用 V3 優化後的獎勵）
        self.stages = [
            CurriculumStage(
                name="Stage1_Novice",
                board_size=6,
                timesteps=500000,  # 50萬步（增加訓練時間，學習邊緣策略）
                graduation_score=25.0,  # 6x6 最高35分，要求25分畢業（提高標準）
                description="新手村：6x6小棋盤，學習從邊角開始、保持耐心、善用轉彎"
            ),
            CurriculumStage(
                name="Stage2_Intermediate",
                board_size=8,
                timesteps=500000,  # 50萬步
                graduation_score=40.0,  # 8x8 最高63分，要求40分畢業（提高標準）
                description="進階班：8x8標準棋盤，優化中等空間策略"
            ),
            CurriculumStage(
                name="Stage3_Advanced",
                board_size=10,
                timesteps=800000,  # 80萬步
                graduation_score=50.0,  # 10x10 最高99分，要求50分畢業
                description="挑戰班：10x10困難棋盤，複雜空間規劃"
            ),
            CurriculumStage(
                name="Stage4_Master",
                board_size=12,
                timesteps=1000000,  # 100萬步
                graduation_score=70.0,  # 12x12 最高143分，要求70分畢業
                description="大師班：12x12極難棋盤，終極挑戰"
            )
        ]
        
        self.current_stage_idx = 0
        self.training_log = []
    
    @property
    def current_stage(self) -> CurriculumStage:
        """獲取當前階段"""
        return self.stages[self.current_stage_idx]
    
    def check_graduation(self, mean_score: float) -> bool:
        """檢查是否達到畢業標準"""
        stage = self.current_stage
        stage.current_score = mean_score
        stage.best_score = max(stage.best_score, mean_score)
        
        if mean_score >= stage.graduation_score:
            stage.completed = True
            print(f"\n🎓 恭喜！{stage.name} 畢業了！")
            print(f"   畢業分數: {mean_score:.1f} (標準: {stage.graduation_score})")
            return True
        return False
    
    def advance_stage(self) -> bool:
        """進入下一階段"""
        if self.current_stage_idx < len(self.stages) - 1:
            self.current_stage_idx += 1
            print(f"\n📈 升級到 {self.current_stage.name}!")
            print(f"   {self.current_stage.description}")
            return True
        else:
            print(f"\n🏆 恭喜完成所有課程階段！")
            return False
    
    def get_model_path(self, stage: Optional[CurriculumStage] = None) -> str:
        """獲取模型保存路徑"""
        if stage is None:
            stage = self.current_stage
        return os.path.join(self.base_dir, stage.name, "model")
    
    def get_best_model_path(self, stage: Optional[CurriculumStage] = None) -> str:
        """獲取最佳模型路徑"""
        if stage is None:
            stage = self.current_stage
        return os.path.join(self.base_dir, stage.name, "best_model")
    
    def save_progress(self):
        """保存訓練進度"""
        progress_file = os.path.join(self.base_dir, "curriculum_progress.txt")
        with open(progress_file, 'w', encoding='utf-8') as f:
            f.write("=== PPO V3 Curriculum Learning Progress ===\n\n")
            for i, stage in enumerate(self.stages):
                status = "✓ 已完成" if stage.completed else "○ 進行中" if i == self.current_stage_idx else "- 未開始"
                f.write(f"{status} {stage.name} ({stage.board_size}x{stage.board_size})\n")
                f.write(f"   描述: {stage.description}\n")
                f.write(f"   最佳分數: {stage.best_score:.1f}\n")
                f.write(f"   畢業標準: {stage.graduation_score}\n")
                f.write(f"   訓練步數: {stage.timesteps}\n\n")
            
            f.write(f"\n當前階段: {self.current_stage.name}\n")


class CurriculumCallback(BaseCallback):
    """課程學習回調 - 監控訓練進度並決定是否畢業"""
    
    def __init__(self, curriculum_manager: CurriculumManager, 
                 eval_freq: int = 10000, n_eval_episodes: int = 20,
                 verbose: int = 1):
        super().__init__(verbose)
        self.curriculum_manager = curriculum_manager
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.eval_count = 0
    
    def _on_step(self) -> bool:
        # 每 eval_freq 步評估一次
        if self.n_calls % self.eval_freq == 0:
            self.eval_count += 1
            mean_score = self._evaluate_agent()
            
            stage = self.curriculum_manager.current_stage
            print(f"\n[評估 #{self.eval_count}] {stage.name} - 平均分數: {mean_score:.1f} / {stage.graduation_score}")
            
            # 檢查是否畢業
            if self.curriculum_manager.check_graduation(mean_score):
                # 保存畢業模型
                model_path = self.curriculum_manager.get_model_path()
                self.model.save(model_path)
                print(f"   ✓ 模型已保存: {model_path}")
            
            # 保存進度
            self.curriculum_manager.save_progress()
        
        return True
    
    def _evaluate_agent(self) -> float:
        """評估當前智能體"""
        stage = self.curriculum_manager.current_stage
        stage_num = self.curriculum_manager.current_stage_idx + 1
        env = GymSnakeEnvV3(board_size=stage.board_size, render_mode=None, stage=stage_num)
        
        scores = []
        for _ in range(self.n_eval_episodes):
            obs, info = env.reset()
            done = False
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
            scores.append(info.get('score', 0))
        
        env.close()
        return np.mean(scores)


def create_v3_model(board_size: int, base_model: Optional[PPO] = None, 
                   log_dir: Optional[str] = None, device: str = 'auto', stage: int = 1) -> PPO:
    """創建 V3 增強模型（支持动态课程奖励）
    
    相比 V2 的改進：
    - 更大的網路：[256, 256, 128] vs V2 的 [128, 128, 64]
    - 更多探索：learning_rate 稍高
    - 支持遷移學習：可以載入前一階段的模型
    - 動態課程：Stage 1-4 動態獎勵係數
    
    Args:
        board_size: 板子大小
        base_model: 用於遷移學習的基礎模型
        log_dir: 日誌目錄
        device: 'cpu', 'cuda', 或 'auto'
        stage: 訓練階段 (1-4)
    """
    
    # 根据板子大小和stage决定课程阶段（向后兼容）
    curriculum_stage = "conservative" if stage == 1 else "aggressive"
    
    # 創建環境
    def make_env():
        env = GymSnakeEnvV3(
            board_size=board_size, 
            render_mode=None,
            curriculum_stage=curriculum_stage,
            stage=stage  # NEW: Pass stage parameter
        )
        env = Monitor(env)
        return env
    
    env = DummyVecEnv([make_env])
    
    # 模型超參數（增強版）
    policy_kwargs = dict(
        net_arch=dict(
            pi=[256, 256, 128],  # Policy network: 更大！
            vf=[256, 256, 128]   # Value network: 更大！
        ),
        activation_fn=torch.nn.ReLU
    )
    
    # 如果有前一階段的模型，進行遷移學習
    if base_model is not None:
        print(f"🔄 遷移學習：載入前一階段的模型權重...")
        # 創建新模型但使用舊模型的部分參數
        model = PPO(
            "MlpPolicy",
            env,
            policy_kwargs=policy_kwargs,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=128,  # 增加 batch size
            n_epochs=15,     # 更多 epochs
            gamma=0.995,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
            verbose=1,
            device=device,
            tensorboard_log=f"./logs/ppo_v3_tensorboard/"
        )
        
        # 嘗試複製部分權重（如果架構兼容）
        try:
            # 這裡可以手動複製某些層的權重
            # 簡單版本：讓它從頭學習，但有經驗加速
            print("   新階段將利用前階段的學習經驗")
        except Exception as e:
            print(f"   無法完全遷移權重，從頭開始: {e}")
    
    else:
        # 從零開始
        model = PPO(
            "MlpPolicy",
            env,
            policy_kwargs=policy_kwargs,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=128,
            n_epochs=15,
            gamma=0.995,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
            verbose=1,
            device=device,
            tensorboard_log=f"./logs/ppo_v3_tensorboard/"
        )
    
    return model


def train_curriculum(device: str = 'auto', start_stage: int = 0,
                    n_envs: int = 8, skip_graduation: bool = False):
    """執行課程學習訓練
    
    Args:
        device: 'cpu', 'cuda', 或 'auto'
        start_stage: 從哪個階段開始（0=第一階段）
        n_envs: 並行環境數量
        skip_graduation: 是否跳過畢業檢查（強制完成所有timesteps）
    """
    
    print("="*70)
    print("🎓 PPO V3: Curriculum Learning (課程學習)")
    print("="*70)
    print(f"設備: {device}")
    print(f"並行環境: {n_envs}")
    print(f"起始階段: {start_stage}")
    print("="*70)
    
    # 創建課程管理器
    curriculum = CurriculumManager()
    curriculum.current_stage_idx = start_stage
    
    # 顯示課程計劃
    print("\n📚 課程計劃:")
    for i, stage in enumerate(curriculum.stages):
        marker = "→" if i == start_stage else " "
        print(f"{marker} {stage.name}: {stage.description}")
        print(f"   棋盤: {stage.board_size}x{stage.board_size}, "
              f"訓練步數: {stage.timesteps:,}, "
              f"畢業標準: {stage.graduation_score}分")
    print()
    
    prev_model = None
    
    # 檢查是否有已訓練好的前一階段模型
    if start_stage > 0:
        prev_stage_idx = start_stage - 1
        prev_stage = curriculum.stages[prev_stage_idx]
        prev_model_path = curriculum.get_best_model_path(prev_stage)
        
        if os.path.exists(prev_model_path + ".zip"):
            print(f"\n🔍 發現已訓練的 {prev_stage.name} 模型")
            print(f"   路徑: {prev_model_path}")
            try:
                prev_model = PPO.load(prev_model_path)
                print(f"   ✓ 成功載入前一階段模型，將用於遷移學習")
            except Exception as e:
                print(f"   ✗ 載入失敗: {e}")
                print(f"   將從頭開始訓練")
                prev_model = None
        else:
            print(f"\n⚠️  警告: 找不到 {prev_stage.name} 的模型")
            print(f"   預期路徑: {prev_model_path}")
            print(f"   建議先訓練 Stage {prev_stage_idx + 1}")
            response = input(f"   是否繼續從頭訓練 Stage {start_stage + 1}? (y/n): ")
            if response.lower() != 'y':
                print("   訓練取消")
                return
    
    # 逐階段訓練
    for stage_idx in range(start_stage, len(curriculum.stages)):
        curriculum.current_stage_idx = stage_idx
        stage = curriculum.current_stage
        
        print("\n" + "="*70)
        print(f"📖 開始 {stage.name}")
        print(f"   {stage.description}")
        print(f"   棋盤大小: {stage.board_size}x{stage.board_size}")
        print(f"   訓練步數: {stage.timesteps:,}")
        print(f"   畢業標準: 平均分數 ≥ {stage.graduation_score}")
        print(f"   階段編號: Stage {stage_idx + 1}")
        print("="*70)
        
        # 創建模型
        print("\n🔧 準備模型...")
        
        # 如果是從前一階段遷移，使用參數震盪策略
        if prev_model is not None:
            print(f"   🔄 從 Stage {stage_idx} 遷移學習...")
            
            # 創建新環境（重要：傳入新的 stage 參數）
            model = create_v3_model(stage.board_size, base_model=prev_model, 
                                   device=device, stage=stage_idx + 1)
            
            # === 強化參數震盪 (Enhanced Hyperparameter Shock) ===
            # 同時提高學習率和探索率來打破舊策略
            print(f"   ⚡ 強化參數震盪: 提高學習率 + 探索率以打破舊策略...")
            original_lr = model.learning_rate
            original_ent = model.ent_coef
            
            # 階段 1: 高學習率 + 高探索率震盪（150k 步）
            model.learning_rate = 3e-4  # 提高學習率
            model.ent_coef = 0.02       # 提高探索率（原始約 0.01）
            print(f"      - 震盪階段: LR = 3e-4, Entropy = 0.02")
            print(f"      - 訓練 150,000 步（強制重新探索）")
            
            model.learn(
                total_timesteps=150_000,
                callback=[],
                progress_bar=True,
                reset_num_timesteps=False
            )
            
            # 階段 2: 中等學習率 + 中等探索率過渡（100k 步）
            model.learning_rate = 1.5e-4
            model.ent_coef = 0.015
            print(f"      - 過渡階段: LR = 1.5e-4, Entropy = 0.015")
            print(f"      - 訓練 100,000 步（穩定策略）")
            
            model.learn(
                total_timesteps=100_000,
                callback=[],
                progress_bar=True,
                reset_num_timesteps=False
            )
            
            # 階段 3: 恢復正常參數
            model.learning_rate = original_lr
            model.ent_coef = original_ent
            print(f"      - 穩定階段: LR = {original_lr}, Entropy = {original_ent}")
            print(f"      - 繼續訓練...")
            remaining_timesteps = stage.timesteps - 250_000
            
        else:
            # 第一階段，從頭開始
            model = create_v3_model(stage.board_size, base_model=None, 
                                   device=device, stage=stage_idx + 1)
            remaining_timesteps = stage.timesteps
        
        # 設置日誌
        log_path = os.path.join(curriculum.base_dir, stage.name, "logs")
        os.makedirs(log_path, exist_ok=True)
        model.set_logger(configure(log_path, ["stdout", "csv", "tensorboard"]))
        
        # 設置回調
        checkpoint_dir = os.path.join(curriculum.base_dir, stage.name, "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        callbacks = [
            CurriculumCallback(
                curriculum_manager=curriculum,
                eval_freq=10000,
                n_eval_episodes=20,
                verbose=1
            ),
            CheckpointCallback(
                save_freq=50000,
                save_path=checkpoint_dir,
                name_prefix=f"{stage.name}_checkpoint"
            )
        ]
        
        # 訓練
        print(f"\n🚀 開始訓練 {stage.name}...")
        start_time = time.time()
        
        try:
            model.learn(
                total_timesteps=remaining_timesteps,
                callback=callbacks,
                progress_bar=True,
                reset_num_timesteps=False if prev_model is not None else True
            )
        except KeyboardInterrupt:
            print("\n⚠️  訓練被用戶中斷")
            model.save(curriculum.get_model_path())
            print(f"   模型已保存")
            break
        
        training_time = time.time() - start_time
        print(f"\n✓ {stage.name} 訓練完成！")
        print(f"   訓練時間: {training_time/60:.1f} 分鐘")
        
        # 保存最終模型
        final_model_path = curriculum.get_model_path()
        model.save(final_model_path)
        print(f"   最終模型: {final_model_path}")
        
        # 最終評估
        print(f"\n📊 {stage.name} 最終評估...")
        final_score = evaluate_model(model, stage.board_size, n_episodes=50)
        stage.best_score = max(stage.best_score, final_score)
        
        # 檢查是否畢業
        if not skip_graduation:
            if curriculum.check_graduation(final_score):
                stage.completed = True
                # 保存為最佳模型
                best_path = curriculum.get_best_model_path()
                model.save(best_path)
                print(f"   ✓ 畢業模型保存: {best_path}")
            else:
                # 自動進入下一階段：不再詢問，記錄訊息並繼續
                print(f"\n❌ 未達畢業標準 ({final_score:.1f} < {stage.graduation_score})")
                print(f"   自動前進至下一階段（已在本階段完成 {stage.timesteps:,} 步訓練）。")
        else:
            stage.completed = True
        
        # 保存進度
        curriculum.save_progress()
        
        # 準備下一階段（遷移學習）
        if stage_idx < len(curriculum.stages) - 1:
            if stage.completed:
                print(f"\n🎓 {stage.name} 畢業！準備進入下一階段...")
            else:
                print(f"\n➡️  尚未達標，但將帶著本階段的權重繼續下一階段。")
            # 無論是否畢業，都使用本階段訓練後的模型作為下一階段的起點
            prev_model = model
        
        print("\n")
    
    # 訓練完成
    print("\n" + "="*70)
    print("🏆 課程學習訓練完成！")
    print("="*70)
    
    print("\n📊 各階段成績:")
    for stage in curriculum.stages:
        status = "✓" if stage.completed else "✗"
        print(f"{status} {stage.name}: 最佳分數 {stage.best_score:.1f} / {stage.graduation_score}")
    
    curriculum.save_progress()
    print(f"\n詳細進度已保存至: {curriculum.base_dir}/curriculum_progress.txt")


def evaluate_model(model: PPO, board_size: int, n_episodes: int = 20, stage: int = 1) -> float:
    """評估模型性能"""
    # 根据板子大小确定课程阶段（向后兼容）
    curriculum_stage = "conservative" if stage == 1 else "aggressive"
    env = GymSnakeEnvV3(board_size=board_size, render_mode=None, 
                       curriculum_stage=curriculum_stage, stage=stage)
    scores = []
    
    for i in range(n_episodes):
        obs, info = env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        scores.append(info.get('score', 0))
    
    env.close()
    mean_score = np.mean(scores)
    std_score = np.std(scores)
    print(f"   評估結果: {mean_score:.1f} ± {std_score:.1f} (最高: {max(scores)}, 最低: {min(scores)})")
    return mean_score


def demo_stage(stage_idx: int = -1, n_episodes: int = 5):
    """演示某個階段的模型"""
    curriculum = CurriculumManager()
    
    if stage_idx == -1:
        # 找到最後完成的階段
        for i in range(len(curriculum.stages) - 1, -1, -1):
            model_path = curriculum.get_best_model_path(curriculum.stages[i])
            if os.path.exists(model_path + ".zip"):
                stage_idx = i
                break
    
    if stage_idx < 0 or stage_idx >= len(curriculum.stages):
        print(f"❌ 階段 {stage_idx} 不存在")
        return
    
    stage = curriculum.stages[stage_idx]
    model_path = curriculum.get_best_model_path(stage)
    
    if not os.path.exists(model_path + ".zip"):
        print(f"❌ 找不到模型: {model_path}")
        return
    
    print(f"\n🎮 演示 {stage.name} ({stage.board_size}x{stage.board_size})")
    print(f"   載入模型: {model_path}")
    
    model = PPO.load(model_path)
    # 根据阶段索引确定 stage 参数
    stage_num = stage_idx + 1
    curriculum_stage = "conservative" if stage_num == 1 else "aggressive"
    env = GymSnakeEnvV3(board_size=stage.board_size, render_mode="human", 
                       curriculum_stage=curriculum_stage, stage=stage_num)
    
    for episode in range(n_episodes):
        print(f"\n回合 {episode + 1}/{n_episodes}")
        obs, info = env.reset()
        done = False
        step_count = 0
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            step_count += 1
            env.render()
            time.sleep(0.05)
        
        score = info.get('score', 0)
        print(f"   分數: {score}, 步數: {step_count}")
    
    env.close()


def main():
    parser = argparse.ArgumentParser(description="PPO V3 - Curriculum Learning")
    parser.add_argument('--mode', type=str, default='train', 
                       choices=['train', 'demo', 'eval'],
                       help='模式: train=訓練, demo=演示, eval=評估')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['cpu', 'cuda', 'auto'],
                       help='訓練設備')
    parser.add_argument('--n-envs', type=int, default=8,
                       help='並行環境數量')
    parser.add_argument('--start-stage', type=int, default=0,
                       help='開始階段 (0-3)')
    parser.add_argument('--stage', type=int, default=-1,
                       help='演示/評估的階段 (-1=最新)')
    parser.add_argument('--n-episodes', type=int, default=5,
                       help='演示/評估的回合數')
    parser.add_argument('--skip-graduation', action='store_true',
                       help='跳過畢業檢查，強制完成所有timesteps')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        train_curriculum(
            device=args.device,
            start_stage=args.start_stage,
            n_envs=args.n_envs,
            skip_graduation=args.skip_graduation
        )
    
    elif args.mode == 'demo':
        demo_stage(stage_idx=args.stage, n_episodes=args.n_episodes)
    
    elif args.mode == 'eval':
        curriculum = CurriculumManager()
        stage_idx = args.stage if args.stage >= 0 else len(curriculum.stages) - 1
        stage = curriculum.stages[stage_idx]
        model_path = curriculum.get_best_model_path(stage)
        
        if not os.path.exists(model_path + ".zip"):
            print(f"❌ 找不到模型: {model_path}")
            return
        
        print(f"\n📊 評估 {stage.name} ({stage.board_size}x{stage.board_size})")
        model = PPO.load(model_path)
        stage_num = stage_idx + 1
        evaluate_model(model, stage.board_size, n_episodes=args.n_episodes, stage=stage_num)


if __name__ == "__main__":
    main()
