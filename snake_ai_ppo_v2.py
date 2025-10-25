"""
Snake AI using PPO - Version 2 (Enhanced with Body-Aware Collision Avoidance)

Key improvements over v1:
1. Progressive collision penalty - heavier penalty for hitting segments near head
2. Enhanced observation space (16-d) with body proximity awareness
3. Self-trap detection and avoidance
4. Better reward shaping for safer navigation
5. Optimized hyperparameters for collision avoidance

Training approach:
- Uses GymSnakeEnvV2 with enhanced collision penalties
- Longer training with more exploration
- Curriculum learning support (optional)
"""
import os
import argparse
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor

from envs.gym_snake_env_v2 import GymSnakeEnvV2


class CollisionAnalysisCallback(BaseCallback):
    """
    Custom callback to track collision statistics during training.
    """
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.collision_count = 0
        self.food_count = 0
        self.total_episodes = 0
        
    def _on_step(self) -> bool:
        # Check if episode just ended
        for i, done in enumerate(self.locals.get('dones', [])):
            if done:
                self.total_episodes += 1
                info = self.locals['infos'][i]
                
                # Track statistics
                if info.get('score', 0) > 0:
                    self.food_count += info['score']
                
                if self.total_episodes % 100 == 0 and self.verbose > 0:
                    avg_food = self.food_count / 100
                    print(f"\nEpisodes: {self.total_episodes} | "
                          f"Avg Food/Episode: {avg_food:.2f}")
                    self.food_count = 0
        
        return True


def train_ppo_v2(
    board_size: int = 8,
    total_timesteps: int = 500000,
    save_path: str = "models/ppo_snake_v2",
    log_path: str = "logs/ppo_snake_v2",
    n_envs: int = 8,
    learning_rate: float = 3e-4,
    n_steps: int = 2048,
    batch_size: int = 64,
    n_epochs: int = 10,
    ent_coef: float = 0.01,
    verbose: int = 1
):
    """
    Train a PPO agent on the Snake game with enhanced collision avoidance.
    
    Args:
        board_size: Size of the game board (default: 8x8)
        total_timesteps: Total training timesteps
        save_path: Directory to save models
        log_path: Directory for tensorboard logs
        n_envs: Number of parallel environments
        learning_rate: Learning rate for PPO
        n_steps: Number of steps per environment per update
        batch_size: Minibatch size
        n_epochs: Number of epochs per update
        ent_coef: Entropy coefficient for exploration
        verbose: Verbosity level (0: none, 1: info, 2: debug)
    """
    print("=" * 70)
    print("Training Snake AI with PPO V2 (Enhanced Collision Avoidance)")
    print("=" * 70)
    print(f"Board size: {board_size}x{board_size}")
    print(f"Total timesteps: {total_timesteps:,}")
    print(f"Parallel environments: {n_envs}")
    print(f"Learning rate: {learning_rate}")
    print(f"Entropy coefficient: {ent_coef} (exploration)")
    
    # Check CUDA availability
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"  GPU Name: {torch.cuda.get_device_name(0)}")
        print(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    else:
        print("  Warning: CUDA not available, using CPU")
    
    print("\n" + "=" * 70)
    print("Key Features (V2):")
    print("  ✓ Progressive collision penalty (neck: -50, body: -10 to -30)")
    print("  ✓ Body proximity awareness in observation space")
    print("  ✓ Self-trap detection and avoidance")
    print("  ✓ Enhanced reward shaping for safer navigation")
    print("=" * 70 + "\n")
    
    # Create directories
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(log_path, exist_ok=True)
    
    # Create vectorized environment (parallel training)
    env = make_vec_env(
        lambda: GymSnakeEnvV2(board_size=board_size, render_mode=None),
        n_envs=n_envs,
        seed=42
    )
    
    # Create evaluation environment
    eval_env = GymSnakeEnvV2(board_size=board_size, render_mode=None, seed=99)
    eval_env = Monitor(eval_env)
    
    # Enhanced policy architecture for better decision making
    policy_kwargs = dict(
        net_arch=[dict(pi=[128, 128, 64], vf=[128, 128, 64])]  # Deeper network
    )
    
    # Initialize PPO agent with optimized parameters for collision avoidance
    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=0.99,  # Discount factor
        gae_lambda=0.95,  # GAE lambda
        clip_range=0.2,  # PPO clipping
        ent_coef=ent_coef,  # Entropy coefficient for exploration
        vf_coef=0.5,  # Value function coefficient
        max_grad_norm=0.5,  # Gradient clipping
        policy_kwargs=policy_kwargs,
        verbose=verbose,
        tensorboard_log=log_path,
        device='auto'  # 'auto' will use GPU if available, otherwise CPU
    )
    
    # Create callbacks
    # Evaluation callback - evaluate agent every 10k steps
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(save_path, "best_model"),
        log_path=os.path.join(log_path, "eval"),
        eval_freq=max(10000 // n_envs, 1),  # Adjust for n_envs
        deterministic=True,
        render=False,
        verbose=1,
        n_eval_episodes=10
    )
    
    # Checkpoint callback - save model every 50k steps
    checkpoint_callback = CheckpointCallback(
        save_freq=max(50000 // n_envs, 1),  # Adjust for n_envs
        save_path=save_path,
        name_prefix="ppo_snake_v2_checkpoint",
        save_replay_buffer=False,
        save_vecnormalize=False,
    )
    
    # Collision analysis callback
    collision_callback = CollisionAnalysisCallback(verbose=1)
    
    # Train the agent
    print("Starting training...")
    print("Monitor training progress with: tensorboard --logdir=" + log_path)
    print()
    
    model.learn(
        total_timesteps=total_timesteps,
        callback=[eval_callback, checkpoint_callback, collision_callback],
        progress_bar=True
    )
    
    # Save final model
    final_model_path = os.path.join(save_path, "ppo_snake_v2_final")
    model.save(final_model_path)
    print(f"\n{'='*70}")
    print(f"Training complete! Model saved to {final_model_path}")
    print(f"{'='*70}")
    
    # Close environments
    env.close()
    eval_env.close()
    
    return model


def evaluate_model_v2(
    model_path: str,
    board_size: int = 8,
    n_eval_episodes: int = 20,
    render: bool = False,
    verbose: bool = True
):
    """
    Evaluate a trained PPO V2 model.
    
    Args:
        model_path: Path to the saved model
        board_size: Size of the game board
        n_eval_episodes: Number of episodes to evaluate
        render: Whether to render the game
        verbose: Print detailed statistics
    """
    print(f"\nEvaluating model: {model_path}")
    print(f"Board size: {board_size}x{board_size}")
    print(f"Episodes: {n_eval_episodes}")
    
    # Load model
    model = PPO.load(model_path)
    
    # Create environment
    render_mode = "human" if render else None
    env = GymSnakeEnvV2(board_size=board_size, render_mode=render_mode)
    
    # Evaluation loop
    scores = []
    snake_lengths = []
    episode_lengths = []
    near_misses = []
    trap_encounters = []
    
    for episode in range(n_eval_episodes):
        obs, info = env.reset()
        done = False
        episode_reward = 0
        steps = 0
        episode_near_misses = 0
        episode_traps = 0
        
        while not done:
            # Predict action (deterministic for evaluation)
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            
            episode_reward += reward
            steps += 1
            episode_near_misses = info.get('near_miss_count', 0)
            if info.get('is_trapped', False):
                episode_traps += 1
            
            done = terminated or truncated
            
            if render:
                env.render()
                import time
                time.sleep(0.1)
        
        scores.append(info['score'])
        snake_lengths.append(info['snake_length'])
        episode_lengths.append(steps)
        near_misses.append(episode_near_misses)
        trap_encounters.append(episode_traps)
        
        if verbose:
            print(f"Episode {episode+1}/{n_eval_episodes}: "
                  f"Score={info['score']}, "
                  f"Length={info['snake_length']}, "
                  f"Steps={steps}, "
                  f"Reward={episode_reward:.2f}, "
                  f"NearMisses={episode_near_misses}, "
                  f"Traps={episode_traps}")
    
    # Print summary statistics
    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY (PPO V2)")
    print("=" * 70)
    print(f"Average Score: {np.mean(scores):.2f} ± {np.std(scores):.2f}")
    print(f"Max Score: {max(scores)} (Best possible: {board_size**2 - 1})")
    print(f"Min Score: {min(scores)}")
    print(f"Average Snake Length: {np.mean(snake_lengths):.2f}")
    print(f"Average Episode Length: {np.mean(episode_lengths):.1f} steps")
    print(f"Average Near Misses: {np.mean(near_misses):.1f}")
    print(f"Average Trap Encounters: {np.mean(trap_encounters):.1f}")
    print(f"\nCollision Avoidance Performance:")
    print(f"  Episodes with zero near-misses: {sum(1 for x in near_misses if x == 0)}/{n_eval_episodes}")
    print(f"  Success rate (score > 0): {sum(1 for x in scores if x > 0) / n_eval_episodes * 100:.1f}%")
    print("=" * 70)
    
    env.close()
    
    return {
        'scores': scores,
        'snake_lengths': snake_lengths,
        'episode_lengths': episode_lengths,
        'mean_score': np.mean(scores),
        'max_score': max(scores),
        'near_misses': near_misses,
        'trap_encounters': trap_encounters
    }


def demo_trained_model_v2(model_path: str, board_size: int = 8, n_episodes: int = 5):
    """
    Demo a trained V2 model with rendering.
    
    Args:
        model_path: Path to saved model
        board_size: Size of game board
        n_episodes: Number of episodes to demo
    """
    print(f"\nDemonstrating trained model (V2): {model_path}")
    
    evaluate_model_v2(
        model_path=model_path,
        board_size=board_size,
        n_eval_episodes=n_episodes,
        render=True,
        verbose=True
    )


def compare_v1_vs_v2(v1_model_path: str, v2_model_path: str, board_size: int = 8, n_episodes: int = 20):
    """
    Compare performance between V1 and V2 models.
    
    Args:
        v1_model_path: Path to V1 model
        v2_model_path: Path to V2 model
        board_size: Size of game board
        n_episodes: Number of evaluation episodes
    """
    print("\n" + "=" * 70)
    print("COMPARING PPO V1 vs V2")
    print("=" * 70)
    
    print("\n--- Evaluating V1 Model ---")
    from envs.gym_snake_env import GymSnakeEnv
    
    try:
        v1_model = PPO.load(v1_model_path)
        v1_env = GymSnakeEnv(board_size=board_size, render_mode=None)
        
        v1_scores = []
        for _ in range(n_episodes):
            obs, _ = v1_env.reset()
            done = False
            while not done:
                action, _ = v1_model.predict(obs, deterministic=True)
                obs, _, terminated, truncated, info = v1_env.step(action)
                done = terminated or truncated
            v1_scores.append(info['score'])
        
        v1_env.close()
        print(f"V1 Average Score: {np.mean(v1_scores):.2f} ± {np.std(v1_scores):.2f}")
    except Exception as e:
        print(f"Could not evaluate V1 model: {e}")
        v1_scores = None
    
    print("\n--- Evaluating V2 Model ---")
    v2_results = evaluate_model_v2(
        model_path=v2_model_path,
        board_size=board_size,
        n_eval_episodes=n_episodes,
        render=False,
        verbose=False
    )
    
    if v1_scores:
        print("\n" + "=" * 70)
        print("COMPARISON RESULTS")
        print("=" * 70)
        print(f"V1 Average Score: {np.mean(v1_scores):.2f}")
        print(f"V2 Average Score: {v2_results['mean_score']:.2f}")
        improvement = (v2_results['mean_score'] - np.mean(v1_scores)) / np.mean(v1_scores) * 100
        print(f"Improvement: {improvement:+.1f}%")
        print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Train Snake AI using PPO V2 (Enhanced Collision Avoidance)")
    parser.add_argument("--mode", choices=['train', 'eval', 'demo', 'compare'], default='train',
                       help="Mode: train, eval, demo, or compare")
    parser.add_argument("--board-size", type=int, default=8,
                       help="Board size (default: 8)")
    parser.add_argument("--timesteps", type=int, default=500000,
                       help="Total training timesteps (default: 500000)")
    parser.add_argument("--model-path", type=str, default="models/ppo_snake_v2/ppo_snake_v2_final",
                       help="Path to save/load model")
    parser.add_argument("--v1-model-path", type=str, default="models/ppo_snake/ppo_snake_final",
                       help="Path to V1 model (for comparison)")
    parser.add_argument("--n-envs", type=int, default=8,
                       help="Number of parallel environments (default: 8)")
    parser.add_argument("--learning-rate", type=float, default=3e-4,
                       help="Learning rate (default: 3e-4)")
    parser.add_argument("--ent-coef", type=float, default=0.01,
                       help="Entropy coefficient for exploration (default: 0.01)")
    parser.add_argument("--n-episodes", type=int, default=20,
                       help="Number of evaluation episodes (default: 20)")
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        # Train a new V2 model
        train_ppo_v2(
            board_size=args.board_size,
            total_timesteps=args.timesteps,
            save_path=os.path.dirname(args.model_path),
            n_envs=args.n_envs,
            learning_rate=args.learning_rate,
            ent_coef=args.ent_coef
        )
    elif args.mode == 'eval':
        # Evaluate existing V2 model
        evaluate_model_v2(
            model_path=args.model_path,
            board_size=args.board_size,
            n_eval_episodes=args.n_episodes,
            render=False
        )
    elif args.mode == 'demo':
        # Demo with rendering
        demo_trained_model_v2(
            model_path=args.model_path,
            board_size=args.board_size,
            n_episodes=args.n_episodes
        )
    elif args.mode == 'compare':
        # Compare V1 vs V2
        compare_v1_vs_v2(
            v1_model_path=args.v1_model_path,
            v2_model_path=args.model_path,
            board_size=args.board_size,
            n_episodes=args.n_episodes
        )


if __name__ == "__main__":
    main()
