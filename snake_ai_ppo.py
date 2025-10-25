"""
Snake AI using PPO (Proximal Policy Optimization) from stable_baselines3.

This is the simplest approach using:
- PPO algorithm from stable_baselines3
- PyTorch as the backend
- Custom Gym environment wrapper

Training approach:
1. Create Gym environment
2. Initialize PPO agent with default parameters
3. Train for specified timesteps
4. Save and evaluate the model
"""
import os
import argparse
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor

from envs.gym_snake_env import GymSnakeEnv


def train_ppo(
    board_size: int = 8,
    total_timesteps: int = 100000,
    save_path: str = "models/ppo_snake",
    log_path: str = "logs/ppo_snake",
    n_envs: int = 4,
    learning_rate: float = 3e-4,
    verbose: int = 1
):
    """
    Train a PPO agent on the Snake game.
    
    Args:
        board_size: Size of the game board (default: 8x8)
        total_timesteps: Total training timesteps
        save_path: Directory to save models
        log_path: Directory for tensorboard logs
        n_envs: Number of parallel environments
        learning_rate: Learning rate for PPO
        verbose: Verbosity level (0: none, 1: info, 2: debug)
    """
    print("=" * 60)
    print("Training Snake AI with PPO (Proximal Policy Optimization)")
    print("=" * 60)
    print(f"Board size: {board_size}x{board_size}")
    print(f"Total timesteps: {total_timesteps:,}")
    print(f"Parallel environments: {n_envs}")
    print(f"Learning rate: {learning_rate}")
    
    # Check CUDA availability
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"  GPU Name: {torch.cuda.get_device_name(0)}")
        print(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    else:
        print("  Warning: CUDA not available, using CPU")
    print("=" * 60)
    
    # Create directories
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(log_path, exist_ok=True)
    
    # Create vectorized environment (parallel training)
    env = make_vec_env(
        lambda: GymSnakeEnv(board_size=board_size, render_mode=None),
        n_envs=n_envs,
        seed=42
    )
    
    # Create evaluation environment
    eval_env = GymSnakeEnv(board_size=board_size, render_mode=None, seed=99)
    eval_env = Monitor(eval_env)
    
    # Initialize PPO agent with simple default parameters
    model = PPO(
        policy="MlpPolicy",  # Simple Multi-Layer Perceptron policy
        env=env,
        learning_rate=learning_rate,
        n_steps=2048,  # Number of steps to run for each environment per update
        batch_size=64,  # Minibatch size
        n_epochs=10,  # Number of epoch when optimizing the surrogate loss
        gamma=0.99,  # Discount factor
        gae_lambda=0.95,  # Factor for trade-off of bias vs variance for GAE
        clip_range=0.2,  # Clipping parameter for PPO
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
        eval_freq=10000,
        deterministic=True,
        render=False,
        verbose=1
    )
    
    # Checkpoint callback - save model every 50k steps
    checkpoint_callback = CheckpointCallback(
        save_freq=50000,
        save_path=save_path,
        name_prefix="ppo_snake_checkpoint",
        save_replay_buffer=False,
        save_vecnormalize=False,
    )
    
    # Train the agent
    print("\nStarting training...")
    model.learn(
        total_timesteps=total_timesteps,
        callback=[eval_callback, checkpoint_callback],
        progress_bar=True
    )
    
    # Save final model
    final_model_path = os.path.join(save_path, "ppo_snake_final")
    model.save(final_model_path)
    print(f"\nTraining complete! Model saved to {final_model_path}")
    
    # Close environments
    env.close()
    eval_env.close()
    
    return model


def evaluate_model(
    model_path: str,
    board_size: int = 8,
    n_eval_episodes: int = 10,
    render: bool = False,
    verbose: bool = True
):
    """
    Evaluate a trained PPO model.
    
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
    env = GymSnakeEnv(board_size=board_size, render_mode=render_mode)
    
    # Evaluation loop
    scores = []
    snake_lengths = []
    episode_lengths = []
    
    for episode in range(n_eval_episodes):
        obs, info = env.reset()
        done = False
        episode_reward = 0
        steps = 0
        
        while not done:
            # Predict action (deterministic for evaluation)
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            
            episode_reward += reward
            steps += 1
            done = terminated or truncated
            
            if render:
                env.render()
                import time
                time.sleep(0.1)
        
        scores.append(info['score'])
        snake_lengths.append(info['snake_length'])
        episode_lengths.append(steps)
        
        if verbose:
            print(f"Episode {episode+1}/{n_eval_episodes}: "
                  f"Score={info['score']}, "
                  f"Length={info['snake_length']}, "
                  f"Steps={steps}, "
                  f"Reward={episode_reward:.2f}")
    
    # Print summary statistics
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Average Score: {np.mean(scores):.2f} ± {np.std(scores):.2f}")
    print(f"Max Score: {max(scores)} (Best possible: {board_size**2 - 1})")
    print(f"Average Snake Length: {np.mean(snake_lengths):.2f}")
    print(f"Average Episode Length: {np.mean(episode_lengths):.1f} steps")
    print("=" * 60)
    
    env.close()
    
    return {
        'scores': scores,
        'snake_lengths': snake_lengths,
        'episode_lengths': episode_lengths,
        'mean_score': np.mean(scores),
        'max_score': max(scores)
    }


def demo_trained_model(model_path: str, board_size: int = 8, n_episodes: int = 5):
    """
    Demo a trained model with rendering.
    
    Args:
        model_path: Path to saved model
        board_size: Size of game board
        n_episodes: Number of episodes to demo
    """
    print(f"\nDemonstrating trained model: {model_path}")
    
    evaluate_model(
        model_path=model_path,
        board_size=board_size,
        n_eval_episodes=n_episodes,
        render=True,
        verbose=True
    )


def main():
    parser = argparse.ArgumentParser(description="Train Snake AI using PPO")
    parser.add_argument("--mode", choices=['train', 'eval', 'demo'], default='train',
                       help="Mode: train, eval, or demo")
    parser.add_argument("--board-size", type=int, default=8,
                       help="Board size (default: 8)")
    parser.add_argument("--timesteps", type=int, default=100000,
                       help="Total training timesteps (default: 100000)")
    parser.add_argument("--model-path", type=str, default="models/ppo_snake/ppo_snake_final",
                       help="Path to save/load model")
    parser.add_argument("--n-envs", type=int, default=4,
                       help="Number of parallel environments (default: 4)")
    parser.add_argument("--learning-rate", type=float, default=3e-4,
                       help="Learning rate (default: 3e-4)")
    parser.add_argument("--n-episodes", type=int, default=10,
                       help="Number of evaluation episodes (default: 10)")
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        # Train a new model
        train_ppo(
            board_size=args.board_size,
            total_timesteps=args.timesteps,
            save_path=os.path.dirname(args.model_path),
            n_envs=args.n_envs,
            learning_rate=args.learning_rate
        )
    elif args.mode == 'eval':
        # Evaluate existing model
        evaluate_model(
            model_path=args.model_path,
            board_size=args.board_size,
            n_eval_episodes=args.n_episodes,
            render=False
        )
    elif args.mode == 'demo':
        # Demo with rendering
        demo_trained_model(
            model_path=args.model_path,
            board_size=args.board_size,
            n_episodes=args.n_episodes
        )


if __name__ == "__main__":
    main()
