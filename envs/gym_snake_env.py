"""Simple Gymnasium environment for Snake Game compatible with stable_baselines3.

This wrapper provides a minimal Gym interface around SnakeGame for training
with stable_baselines3 algorithms like PPO.

Observation: 12-d binary feature vector (danger detection + food direction + current direction)
Action space: Discrete(4) -> 0=UP, 1=LEFT, 2=RIGHT, 3=DOWN
"""
import numpy as np
import random
from typing import Optional, Tuple

import gymnasium as gym
from gymnasium import spaces

# Import SnakeGame - handle both direct execution and module import
try:
    from envs.snake_game import SnakeGame
except ImportError:
    from snake_game import SnakeGame


class GymSnakeEnv(gym.Env):
    """Gymnasium environment wrapping SnakeGame for stable_baselines3.
    
    Simplified observation space and reward shaping for PPO training.
    """

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(self, board_size: int = 8, render_mode: Optional[str] = None, seed: Optional[int] = None):
        super().__init__()
        
        self.board_size = board_size
        self.render_mode = render_mode
        self.seed_value = seed if seed is not None else random.randint(0, int(1e9))

        # Create the underlying game
        silent_mode = not (render_mode == "human")
        self.game = SnakeGame(seed=self.seed_value, board_size=self.board_size, silent_mode=silent_mode)

        # Define action and observation spaces
        self.action_space = spaces.Discrete(4)  # UP, LEFT, RIGHT, DOWN
        self.observation_space = spaces.Box(low=0, high=1, shape=(12,), dtype=np.float32)

        # Track episode statistics
        self.episode_length = 0
        self.max_episode_length = board_size * board_size * 10  # Allow longer episodes for better play

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, dict]:
        """Reset the environment to initial state."""
        if seed is not None:
            self.seed_value = seed
            random.seed(seed)
            np.random.seed(seed)
            
        self.game.reset()
        self.episode_length = 0
        
        obs = self._get_obs()
        info = self._get_info()
        
        return obs, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """Execute one step in the environment."""
        self.episode_length += 1
        
        # Execute action in the game
        done, game_info = self.game.step(int(action))
        
        # Get new observation
        obs = self._get_obs()
        
        # Calculate reward
        reward = self._calculate_reward(done, game_info)
        
        # Check if episode should terminate
        terminated = done
        truncated = self.episode_length >= self.max_episode_length
        
        # Combine info
        info = self._get_info()
        info.update(game_info)
        
        return obs, reward, terminated, truncated, info

    def render(self):
        """Render the environment."""
        if self.render_mode == "human":
            if hasattr(self.game, 'render'):
                self.game.render()
        return None

    def close(self):
        """Clean up resources."""
        pass

    def _get_obs(self) -> np.ndarray:
        """Compute observation from current game state.
        
        Returns 12-d binary feature vector:
        - 4 danger detection (up, down, left, right)
        - 4 food direction (up, down, left, right)
        - 4 current direction (one-hot encoding)
        """
        if not self.game.snake or len(self.game.snake) == 0:
            return np.zeros(12, dtype=np.float32)

        head_row, head_col = self.game.snake[0]
        food_row, food_col = self.game.food

        # Danger detection (wall or body collision)
        danger_up = int((head_row - 1 < 0) or ((head_row - 1, head_col) in self.game.snake))
        danger_down = int((head_row + 1 >= self.board_size) or ((head_row + 1, head_col) in self.game.snake))
        danger_left = int((head_col - 1 < 0) or ((head_row, head_col - 1) in self.game.snake))
        danger_right = int((head_col + 1 >= self.board_size) or ((head_row, head_col + 1) in self.game.snake))

        # Food direction
        food_up = int(food_row < head_row)
        food_down = int(food_row > head_row)
        food_left = int(food_col < head_col)
        food_right = int(food_col > head_col)

        # Current direction (one-hot)
        dir_up = int(self.game.direction == "UP")
        dir_down = int(self.game.direction == "DOWN")
        dir_left = int(self.game.direction == "LEFT")
        dir_right = int(self.game.direction == "RIGHT")

        obs = np.array([
            danger_up, danger_down, danger_left, danger_right,
            food_up, food_down, food_left, food_right,
            dir_up, dir_down, dir_left, dir_right
        ], dtype=np.float32)

        return obs

    def _calculate_reward(self, done: bool, info: dict) -> float:
        """Calculate reward for the current step.
        
        Reward shaping:
        - Food eaten: +10
        - Game over (collision): -10
        - Win (fill board): +50
        - Step survival: +0.1
        - Moving towards food: +0.5
        - Moving away from food: -0.5
        """
        reward = 0.0

        if done:
            # Check if won (snake fills entire board)
            if len(self.game.snake) >= self.game.grid_size:
                reward = 50.0  # Big reward for winning
            else:
                reward = -10.0  # Penalty for collision
        elif info.get('food_obtained', False):
            # Reward for eating food
            reward = 10.0
        else:
            # Small reward for staying alive
            reward = 0.1
            
            # Additional reward shaping based on distance to food
            head_pos = info.get('snake_head_pos')
            prev_pos = info.get('prev_snake_head_pos')
            food_pos = info.get('food_pos')
            
            if head_pos is not None and prev_pos is not None and food_pos is not None:
                prev_distance = np.abs(prev_pos - food_pos).sum()
                current_distance = np.abs(head_pos - food_pos).sum()
                
                if current_distance < prev_distance:
                    reward += 0.5  # Moved closer to food
                elif current_distance > prev_distance:
                    reward -= 0.5  # Moved away from food

        return float(reward)

    def _get_info(self) -> dict:
        """Get additional info about the environment state."""
        return {
            'score': self.game.score,
            'snake_length': len(self.game.snake) if self.game.snake else 0,
            'episode_length': self.episode_length,
        }


# Register environment with gymnasium
try:
    from gymnasium.envs.registration import register
    
    register(
        id='Snake-v0',
        entry_point='envs.gym_snake_env:GymSnakeEnv',
        max_episode_steps=1000,
    )
except Exception:
    pass  # Already registered or registration failed


if __name__ == "__main__":
    # Quick test of the environment
    print("Testing GymSnakeEnv...")
    env = GymSnakeEnv(board_size=8, render_mode=None, seed=42)
    
    obs, info = env.reset()
    print(f"Initial observation shape: {obs.shape}")
    print(f"Initial info: {info}")
    
    total_reward = 0
    for step in range(100):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        if terminated or truncated:
            print(f"Episode finished after {step+1} steps")
            print(f"Final score: {info['score']}")
            print(f"Total reward: {total_reward:.2f}")
            break
    else:
        print(f"Episode did not finish in 100 steps, total reward: {total_reward:.2f}")
