"""
Enhanced Gymnasium environment for Snake Game (Version 2).

Key improvements over v1:
1. Body proximity penalty - heavier penalty for hitting segments closer to head
2. Self-trap detection - penalty for getting trapped by own body
3. Enhanced observation space - includes body proximity awareness
4. Better reward shaping for safer navigation

Observation: 16-d feature vector (danger + body proximity + food direction + current direction)
Action space: Discrete(4) -> 0=UP, 1=LEFT, 2=RIGHT, 3=DOWN
"""
import numpy as np
import random
from typing import Optional, Tuple, List

import gymnasium as gym
from gymnasium import spaces

# Import SnakeGame
try:
    from envs.snake_game import SnakeGame
except ImportError:
    from snake_game import SnakeGame


class GymSnakeEnvV2(gym.Env):
    """Enhanced Gymnasium environment with body-aware collision penalties.
    
    Main improvements:
    - Progressive penalty based on collision segment proximity to head
    - Body proximity awareness in observation space
    - Self-trap detection and avoidance incentives
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
        
        # Enhanced observation: 16 features
        # [0-3]: danger detection (up, down, left, right)
        # [4-7]: body proximity in each direction (normalized distance to nearest body segment)
        # [8-11]: food direction (up, down, left, right)
        # [12-15]: current direction (one-hot)
        self.observation_space = spaces.Box(low=0, high=1, shape=(16,), dtype=np.float32)

        # Track episode statistics
        self.episode_length = 0
        self.max_episode_length = board_size * board_size * 3  # More lenient for exploration
        
        # Previous position for reward shaping
        self.prev_head_pos = None
        self.prev_distance_to_food = None
        
        # Track self-collision attempts
        self.near_miss_count = 0

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, dict]:
        """Reset the environment to initial state."""
        if seed is not None:
            self.seed_value = seed
            random.seed(seed)
            np.random.seed(seed)
            
        self.game.reset()
        self.episode_length = 0
        self.near_miss_count = 0
        
        # Initialize tracking variables
        if self.game.snake:
            self.prev_head_pos = np.array(self.game.snake[0])
            self.prev_distance_to_food = self._manhattan_distance(
                self.game.snake[0], self.game.food
            )
        
        obs = self._get_obs()
        info = self._get_info()
        
        return obs, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """Execute one step in the environment."""
        self.episode_length += 1
        
        # Store previous state
        prev_snake_length = len(self.game.snake)
        prev_head = self.game.snake[0] if self.game.snake else None
        
        # Execute action in the game
        done, game_info = self.game.step(int(action))
        
        # Get new observation
        obs = self._get_obs()
        
        # Calculate enhanced reward
        reward = self._calculate_reward_v2(done, game_info, prev_head, prev_snake_length)
        
        # Check if episode should terminate
        terminated = done
        truncated = self.episode_length >= self.max_episode_length
        
        # Update tracking variables
        if not done and self.game.snake:
            self.prev_head_pos = np.array(self.game.snake[0])
            self.prev_distance_to_food = self._manhattan_distance(
                self.game.snake[0], self.game.food
            )
        
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
        """Compute enhanced observation from current game state.
        
        Returns 16-d feature vector:
        - 4 danger detection (immediate collision risk)
        - 4 body proximity (distance to nearest body segment in each direction)
        - 4 food direction (up, down, left, right)
        - 4 current direction (one-hot encoding)
        """
        if not self.game.snake or len(self.game.snake) == 0:
            return np.zeros(16, dtype=np.float32)

        head_row, head_col = self.game.snake[0]
        food_row, food_col = self.game.food

        # 1. Danger detection (immediate collision)
        danger_up = int((head_row - 1 < 0) or ((head_row - 1, head_col) in self.game.snake))
        danger_down = int((head_row + 1 >= self.board_size) or ((head_row + 1, head_col) in self.game.snake))
        danger_left = int((head_col - 1 < 0) or ((head_row, head_col - 1) in self.game.snake))
        danger_right = int((head_col + 1 >= self.board_size) or ((head_row, head_col + 1) in self.game.snake))

        # 2. Body proximity (normalized distance to nearest body segment)
        body_prox_up, body_prox_down, body_prox_left, body_prox_right = self._get_body_proximity(head_row, head_col)

        # 3. Food direction
        food_up = int(food_row < head_row)
        food_down = int(food_row > head_row)
        food_left = int(food_col < head_col)
        food_right = int(food_col > head_col)

        # 4. Current direction (one-hot)
        dir_up = int(self.game.direction == "UP")
        dir_down = int(self.game.direction == "DOWN")
        dir_left = int(self.game.direction == "LEFT")
        dir_right = int(self.game.direction == "RIGHT")

        obs = np.array([
            # Danger (4)
            danger_up, danger_down, danger_left, danger_right,
            # Body proximity (4)
            body_prox_up, body_prox_down, body_prox_left, body_prox_right,
            # Food direction (4)
            food_up, food_down, food_left, food_right,
            # Current direction (4)
            dir_up, dir_down, dir_left, dir_right
        ], dtype=np.float32)

        return obs

    def _get_body_proximity(self, head_row: int, head_col: int) -> Tuple[float, float, float, float]:
        """Calculate normalized distance to nearest body segment in each direction.
        
        Returns values between 0 and 1, where:
        - 1.0 = no body segment in that direction
        - 0.0 = body segment immediately adjacent
        """
        max_distance = self.board_size
        
        # Search for nearest body segment in each direction
        body_segments = self.game.snake[1:]  # Exclude head
        
        # Up direction
        up_dist = max_distance
        for seg_row, seg_col in body_segments:
            if seg_col == head_col and seg_row < head_row:
                up_dist = min(up_dist, head_row - seg_row)
        
        # Down direction
        down_dist = max_distance
        for seg_row, seg_col in body_segments:
            if seg_col == head_col and seg_row > head_row:
                down_dist = min(down_dist, seg_row - head_row)
        
        # Left direction
        left_dist = max_distance
        for seg_row, seg_col in body_segments:
            if seg_row == head_row and seg_col < head_col:
                left_dist = min(left_dist, head_col - seg_col)
        
        # Right direction
        right_dist = max_distance
        for seg_row, seg_col in body_segments:
            if seg_row == head_row and seg_col > head_col:
                right_dist = min(right_dist, seg_col - head_col)
        
        # Normalize to [0, 1] range
        prox_up = up_dist / max_distance
        prox_down = down_dist / max_distance
        prox_left = left_dist / max_distance
        prox_right = right_dist / max_distance
        
        return prox_up, prox_down, prox_left, prox_right

    def _calculate_reward_v2(self, done: bool, info: dict, prev_head: Optional[Tuple], 
                             prev_snake_length: int) -> float:
        """Enhanced reward calculation with body-aware penalties.
        
        Reward structure:
        - Food eaten: +10
        - Win (fill board): +100
        - Collision with body: -10 to -50 (based on segment proximity to head)
        - Collision with wall: -20
        - Moving towards food: +1.0
        - Moving away from food: -0.5
        - Near-miss with body (close call): -1.0
        - Survival: +0.1
        - Trap detection: -2.0
        """
        reward = 0.0

        if done:
            # Check if won (snake fills entire board)
            if len(self.game.snake) >= self.game.grid_size:
                reward = 100.0  # Huge reward for winning
            else:
                # Determine collision type and severity
                if prev_head is not None:
                    reward = self._calculate_collision_penalty(prev_head)
                else:
                    reward = -20.0  # Default collision penalty
        
        elif info.get('food_obtained', False):
            # Reward for eating food
            reward = 10.0
            # Bonus for longer snake (encourages growth)
            snake_length = len(self.game.snake)
            reward += min(snake_length * 0.5, 5.0)  # Cap bonus at +5
        
        else:
            # Alive and moving
            reward = 0.1
            
            # Distance-based reward shaping (stronger signal)
            if self.game.snake and self.game.food:
                current_distance = self._manhattan_distance(self.game.snake[0], self.game.food)
                
                if self.prev_distance_to_food is not None:
                    if current_distance < self.prev_distance_to_food:
                        reward += 1.0  # Good! Moving towards food
                    elif current_distance > self.prev_distance_to_food:
                        reward -= 0.5  # Bad! Moving away from food
            
            # Check for near-misses with body (learning to avoid risky moves)
            if self._is_near_body():
                reward -= 1.0
                self.near_miss_count += 1
            
            # Penalty for getting trapped (no safe moves)
            if self._is_trapped():
                reward -= 2.0

        return float(reward)

    def _calculate_collision_penalty(self, prev_head: Tuple[int, int]) -> float:
        """Calculate penalty based on what the snake collided with.
        
        Body collision penalty scales with proximity to head:
        - Segment 1 (neck): -50
        - Segment 2-3: -30
        - Segment 4-5: -20
        - Segment 6+: -10
        
        Wall collision: -20
        """
        head_row, head_col = prev_head
        
        # Check for wall collision
        direction = self.game.direction
        if direction == "UP" and head_row == 0:
            return -20.0
        elif direction == "DOWN" and head_row == self.board_size - 1:
            return -20.0
        elif direction == "LEFT" and head_col == 0:
            return -20.0
        elif direction == "RIGHT" and head_col == self.board_size - 1:
            return -20.0
        
        # Must be body collision - find which segment
        # Try to determine collision position
        collision_pos = None
        if direction == "UP":
            collision_pos = (head_row - 1, head_col)
        elif direction == "DOWN":
            collision_pos = (head_row + 1, head_col)
        elif direction == "LEFT":
            collision_pos = (head_row, head_col - 1)
        elif direction == "RIGHT":
            collision_pos = (head_row, head_col + 1)
        
        if collision_pos is None:
            return -20.0  # Unknown collision
        
        # Find which body segment was hit
        try:
            segment_index = self.game.snake.index(collision_pos)
            
            # Progressive penalty based on proximity to head
            if segment_index == 1:  # Hit the neck (most dangerous!)
                return -50.0
            elif segment_index <= 3:  # Hit segments 2-3
                return -30.0
            elif segment_index <= 5:  # Hit segments 4-5
                return -20.0
            else:  # Hit segments 6+
                return -10.0
        except (ValueError, AttributeError):
            # Segment not found, default penalty
            return -20.0

    def _is_near_body(self) -> bool:
        """Check if head is adjacent to body (excluding neck)."""
        if not self.game.snake or len(self.game.snake) < 4:
            return False
        
        head_row, head_col = self.game.snake[0]
        body_segments = self.game.snake[2:]  # Exclude head and neck
        
        # Check all adjacent positions
        adjacent_positions = [
            (head_row - 1, head_col),  # Up
            (head_row + 1, head_col),  # Down
            (head_row, head_col - 1),  # Left
            (head_row, head_col + 1),  # Right
        ]
        
        for pos in adjacent_positions:
            if pos in body_segments:
                return True
        
        return False

    def _is_trapped(self) -> bool:
        """Check if the snake is in a trapped situation (no safe moves).
        
        Returns True if all 4 directions lead to immediate collision.
        """
        if not self.game.snake:
            return False
        
        head_row, head_col = self.game.snake[0]
        
        # Check all 4 directions
        safe_moves = 0
        
        # Up
        if head_row > 0 and (head_row - 1, head_col) not in self.game.snake:
            safe_moves += 1
        
        # Down
        if head_row < self.board_size - 1 and (head_row + 1, head_col) not in self.game.snake:
            safe_moves += 1
        
        # Left
        if head_col > 0 and (head_row, head_col - 1) not in self.game.snake:
            safe_moves += 1
        
        # Right
        if head_col < self.board_size - 1 and (head_row, head_col + 1) not in self.game.snake:
            safe_moves += 1
        
        return safe_moves == 0

    def _manhattan_distance(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> float:
        """Calculate Manhattan distance between two positions."""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def _get_info(self) -> dict:
        """Get additional info about the environment state."""
        return {
            'score': self.game.score,
            'snake_length': len(self.game.snake) if self.game.snake else 0,
            'episode_length': self.episode_length,
            'near_miss_count': self.near_miss_count,
            'is_trapped': self._is_trapped() if self.game.snake else False,
        }


if __name__ == "__main__":
    # Quick test of the enhanced environment
    print("Testing GymSnakeEnvV2...")
    env = GymSnakeEnvV2(board_size=8, render_mode=None, seed=42)
    
    obs, info = env.reset()
    print(f"Initial observation shape: {obs.shape}")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    print(f"Initial info: {info}")
    
    total_reward = 0
    for step in range(100):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        if reward < -5:
            print(f"Step {step}: Large penalty! Reward={reward:.2f}")
        
        if terminated or truncated:
            print(f"\nEpisode finished after {step+1} steps")
            print(f"Final score: {info['score']}")
            print(f"Final snake length: {info['snake_length']}")
            print(f"Near misses: {info['near_miss_count']}")
            print(f"Total reward: {total_reward:.2f}")
            break
    else:
        print(f"\nEpisode did not finish in 100 steps")
        print(f"Total reward: {total_reward:.2f}")
    
    print("\n✓ Environment test completed!")
