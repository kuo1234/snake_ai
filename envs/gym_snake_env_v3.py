"""
Curriculum-optimized Gymnasium environment for Snake Game (Version 3).

Dynamic Reward Curriculum:
- Stage 1 (6x6): Conservative strategy - edge preference, patience, safety first
- Stage 2+ (8x8+): Aggressive strategy - food-seeking, reduced edge bonus, hunger penalty

Key improvements for small board (6x6) training:
1. Edge-aware rewards - encourages peripheral movement (Stage 1 only)
2. Patience rewards - slower, safer approach to food (Stage 1 only)
3. Turn efficiency - rewards strategic turning to avoid body
4. Space management - penalizes getting boxed in center
5. Progressive difficulty - adapts rewards based on snake length
6. Hunger timer - penalizes lack of progress (Stage 2+ only)

Observation: 20-d feature vector (enhanced spatial awareness)
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


class GymSnakeEnvV3(gym.Env):
    """Curriculum-optimized environment with dynamic reward strategy.
    
    Main improvements:
    - Dynamic reward curriculum based on board size
    - Stage 1 (6x6): Edge preference, patience (conservative)
    - Stage 2+ (8x8+): Food-seeking, hunger timer (aggressive)
    - Turn efficiency tracking
    - Space management awareness
    """

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(self, board_size: int = 6, render_mode: Optional[str] = None, 
                 seed: Optional[int] = None, curriculum_stage: str = "conservative",
                 stage: int = 1):
        """
        Args:
            board_size: Size of the game board
            render_mode: Rendering mode (None, "human", "rgb_array")
            seed: Random seed
            curriculum_stage: "conservative" (Stage 1) or "aggressive" (Stage 2+) [DEPRECATED - use stage instead]
            stage: Training stage (1-4) - determines dynamic reward coefficients
        """
        super().__init__()
        
        self.board_size = board_size
        self.render_mode = render_mode
        self.seed_value = seed if seed is not None else random.randint(0, int(1e9))
        
        # Curriculum stage determines reward strategy (backward compatibility)
        self.curriculum_stage = curriculum_stage
        
        # New: Stage-based dynamic rewards
        self.stage = stage
        self.hunger_timer = 0
        self.hunger_limit = self.board_size * self.board_size

        # Create the underlying game
        silent_mode = not (render_mode == "human")
        self.game = SnakeGame(seed=self.seed_value, board_size=self.board_size, silent_mode=silent_mode)

        # Define action and observation spaces
        self.action_space = spaces.Discrete(4)  # UP, LEFT, RIGHT, DOWN
        
        # Enhanced observation: 20 features
        # [0-3]: danger detection (up, down, left, right)
        # [4-7]: body proximity in each direction
        # [8-11]: food direction (up, down, left, right)
        # [12-15]: current direction (one-hot)
        # [16-17]: edge proximity (distance to nearest wall, normalized)
        # [18]: snake length ratio (current_length / max_possible)
        # [19]: available space ratio (empty cells / total cells)
        self.observation_space = spaces.Box(low=0, high=1, shape=(20,), dtype=np.float32)

        # Track episode statistics
        self.episode_length = 0
        self.max_episode_length = board_size * board_size * 15  # More patience for small boards
        
        # Previous position for reward shaping
        self.prev_head_pos = None
        self.prev_distance_to_food = None
        self.prev_direction = None
        
        # Track strategic behaviors
        self.turn_count = 0
        self.edge_time = 0
        self.successful_turns = 0  # Turns that avoided body collision
        
        # Hunger timer (Stage 2+ only)
        self.steps_since_food = 0
        self.hunger_threshold = board_size * board_size  # e.g., 64 for 8x8
        
        # Curriculum-aware difficulty scaling
        self.difficulty_level = "easy"  # easy -> medium -> hard
        
        # Print curriculum mode on init
        if curriculum_stage == "conservative":
            print(f"  🐢 保守模式 (Stage 1): 边缘策略，耐心觅食")
        else:
            print(f"  🏃 积极模式 (Stage 2+): 主动追食，饥饿惩罚")

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None, 
              stage: Optional[int] = None) -> Tuple[np.ndarray, dict]:
        """Reset the environment to initial state.
        
        Args:
            seed: Random seed
            options: Additional options
            stage: Optional stage to switch to (updates reward coefficients dynamically)
        """
        if seed is not None:
            self.seed_value = seed
            random.seed(seed)
            np.random.seed(seed)
        
        # Dynamic stage switching
        if stage is not None:
            self.stage = stage
            self.hunger_limit = self.board_size * self.board_size
            
        self.game.reset()
        self.episode_length = 0
        self.turn_count = 0
        self.edge_time = 0
        self.successful_turns = 0
        self.steps_since_food = 0  # Reset hunger timer
        self.hunger_timer = 0  # Reset new hunger timer
        
        # Initialize tracking variables
        if self.game.snake:
            self.prev_head_pos = np.array(self.game.snake[0])
            self.prev_distance_to_food = self._manhattan_distance(
                self.game.snake[0], self.game.food
            )
            self.prev_direction = self.game.direction
        
        # Adjust difficulty based on board size
        if self.board_size <= 6:
            self.difficulty_level = "easy"
        elif self.board_size <= 10:
            self.difficulty_level = "medium"
        else:
            self.difficulty_level = "hard"
        
        obs = self._get_obs()
        info = self._get_info()
        
        return obs, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """Execute one step in the environment."""
        self.episode_length += 1
        
        # Hunger timer (all stages)
        self.hunger_timer += 1
        
        # Convert action to int if it's a numpy array
        if isinstance(action, np.ndarray):
            action = int(action.item())
        else:
            action = int(action)
        
        # Store previous state
        prev_snake_length = len(self.game.snake)
        prev_head = self.game.snake[0] if self.game.snake else None
        prev_direction = self.game.direction
        
        # Track if this is a turn
        action_to_direction = {0: "UP", 1: "LEFT", 2: "RIGHT", 3: "DOWN"}
        new_direction = action_to_direction.get(action, prev_direction)
        is_turn = (new_direction != prev_direction)
        
        # Execute action in the game (action already converted to int above)
        done, game_info = self.game.step(action)
        
        # Check if food was eaten (reset hunger timer)
        if len(self.game.snake) > prev_snake_length:
            self.hunger_timer = 0
            self.steps_since_food = 0
        
        # Get new observation
        obs = self._get_obs()
        
        # Calculate curriculum-optimized reward
        reward = self._calculate_reward_v3(
            done, game_info, prev_head, prev_snake_length, 
            is_turn, prev_direction
        )
        
        # Check if episode should terminate
        terminated = done
        truncated = self.episode_length >= self.max_episode_length
        
        # Update tracking variables
        if not done and self.game.snake:
            self.prev_head_pos = np.array(self.game.snake[0])
            self.prev_distance_to_food = self._manhattan_distance(
                self.game.snake[0], self.game.food
            )
            self.prev_direction = self.game.direction
            
            # Track edge time
            if self._is_on_edge():
                self.edge_time += 1
            
            # Track successful turns
            if is_turn and not self._is_near_body():
                self.successful_turns += 1
        
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
        
        Returns 20-d feature vector with spatial awareness.
        """
        if not self.game.snake or len(self.game.snake) == 0:
            return np.zeros(20, dtype=np.float32)

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

        # 5. Edge proximity (normalized distance to nearest wall)
        edge_dist_vertical = min(head_row, self.board_size - 1 - head_row) / (self.board_size / 2)
        edge_dist_horizontal = min(head_col, self.board_size - 1 - head_col) / (self.board_size / 2)

        # 6. Snake length ratio
        snake_length_ratio = len(self.game.snake) / self.game.grid_size

        # 7. Available space ratio
        occupied_cells = len(self.game.snake)
        available_space_ratio = (self.game.grid_size - occupied_cells) / self.game.grid_size

        obs = np.array([
            # Danger (4)
            danger_up, danger_down, danger_left, danger_right,
            # Body proximity (4)
            body_prox_up, body_prox_down, body_prox_left, body_prox_right,
            # Food direction (4)
            food_up, food_down, food_left, food_right,
            # Current direction (4)
            dir_up, dir_down, dir_left, dir_right,
            # Spatial awareness (4)
            edge_dist_vertical, edge_dist_horizontal,
            snake_length_ratio, available_space_ratio
        ], dtype=np.float32)

        return obs

    def _get_body_proximity(self, head_row: int, head_col: int) -> Tuple[float, float, float, float]:
        """Calculate normalized distance to nearest body segment in each direction."""
        max_distance = self.board_size
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

    def _calculate_reward_v3(self, done: bool, info: dict, prev_head: Optional[Tuple], 
                             prev_snake_length: int, is_turn: bool, prev_direction: str) -> float:
        """Dynamic stage-based reward system (V3 with Stage 1-4 coefficients).
        
        Stage 1 (6x6) - Conservative:
        - Approach: +0.3, Away: -0.2, Survival: +0.2
        - Edge bonus: +0.3, Center: +0.0, Trap: -1.5, Wall: -10.0
        
        Stage 2 (8x8) - Aggressive:
        - Approach: +1.0, Away: -0.5, Survival: +0.1
        - Edge bonus: +0.0, Center: +0.5, Trap: -3.0, Wall: -20.0
        - Hunger timer: -10.0 if no food in board_size^2 steps
        
        Stage 3 (10x10) - Advanced:
        - Approach: +1.0, Away: -0.5, Survival: +0.1
        - Edge bonus: +0.0, Center: +1.0, Trap: -5.0, Wall: -20.0
        - Hunger timer: -10.0
        
        Stage 4 (12x12) - Master:
        - Approach: +0.5, Away: -0.3, Survival: +0.2
        - Edge bonus: +0.0, Center: +1.0, Trap: -5.0, Wall: -20.0
        - Hunger timer: -10.0
        """
        reward = 0.0
        snake_length = len(self.game.snake) if self.game.snake else prev_snake_length
        
        # Stage-based dynamic reward coefficients
        reward_config = {
            1: {  # Stage 1 (6x6) - Conservative
                'approach': 0.3,
                'away': -0.2,
                'survival': 0.2,
                'edge_bonus': 0.3,
                'center_open': 0.0,
                'trap_penalty': -1.5,
                'wall_penalty': -10.0,
                'hunger_enabled': False
            },
            2: {  # Stage 2 (8x8) - Aggressive
                'approach': 1.0,
                'away': -0.5,
                'survival': 0.1,
                'edge_bonus': 0.0,
                'center_open': 0.5,
                'trap_penalty': -10.0,  # 从 -3.0 提高到 -10.0
                'wall_penalty': -20.0,
                'hunger_enabled': True
            },
            3: {  # Stage 3 (10x10) - Advanced
                'approach': 1.0,
                'away': -0.5,
                'survival': 0.1,
                'edge_bonus': 0.0,
                'center_open': 1.0,
                'trap_penalty': -15.0,  # 从 -5.0 提高到 -15.0
                'wall_penalty': -20.0,
                'hunger_enabled': True
            },
            4: {  # Stage 4 (12x12) - Master
                'approach': 0.5,
                'away': -0.3,
                'survival': 0.2,
                'edge_bonus': 0.0,
                'center_open': 1.0,
                'trap_penalty': -15.0,  # 从 -5.0 提高到 -15.0
                'wall_penalty': -20.0,
                'hunger_enabled': True
            }
        }
        
        # Get config for current stage (default to stage 1)
        config = reward_config.get(self.stage, reward_config[1])

        if done:
            # Check if won
            if snake_length >= self.game.grid_size:
                reward = 100.0
            else:
                # Stage-based death penalty
                reward = config['wall_penalty'] if self._is_wall_collision(prev_head) else config['trap_penalty']
        
        elif info.get('food_obtained', False):
            # Reward for eating food (fixed across all stages)
            reward = 15.0
            
            # Progressive bonus based on snake length (encourages growth)
            length_bonus = min(snake_length * 0.3, 4.0)
            reward += length_bonus
            
            # Extra bonus for eating while on edge (Stage 1 only)
            if config['edge_bonus'] > 0 and self._is_on_edge():
                reward += 2.0
            
            # Reset hunger timer
            self.steps_since_food = 0
        
        else:
            # Base survival reward (stage-dependent)
            reward = config['survival']
            
            # === Edge Movement Bonus (Stage 1 only) ===
            if config['edge_bonus'] > 0 and self._is_on_edge():
                # Stronger edge bonus when snake is short
                edge_multiplier = max(1.0, (1.5 - snake_length / self.game.grid_size))
                reward += config['edge_bonus'] * edge_multiplier
            
            # === Distance-based reward (dynamic urgency) ===
            if self.game.snake and self.game.food:
                current_distance = self._manhattan_distance(self.game.snake[0], self.game.food)
                
                if self.prev_distance_to_food is not None:
                    if current_distance < self.prev_distance_to_food:
                        # Moving towards food
                        reward += config['approach']
                    elif current_distance > self.prev_distance_to_food:
                        # Moving away from food
                        reward += config['away']  # Note: this is negative
            
            # === Tactical Turns (all stages) ===
            if is_turn:
                self.turn_count += 1
                
                # Check if this turn avoided a body collision
                if self._would_have_collided(prev_direction):
                    # Smart turn! Reward it
                    reward += 0.5
                    self.successful_turns += 1
            
            # Penalty for risky moves near body (all stages)
            if self._is_near_body():
                reward -= 0.3
            
            # === Space Management (stage-dependent) ===
            if snake_length > self.game.grid_size * 0.3:
                if not self._is_center_crowded() and config['center_open'] > 0:
                    reward += config['center_open']
            
            # Trap penalty (stage-dependent)
            if self._is_trapped():
                reward += config['trap_penalty']  # Note: this is negative
        
        # === Hunger Timer Penalty (Stage 2+ only) ===
        if config['hunger_enabled'] and self.hunger_timer > self.hunger_limit:
            reward -= 10.0
            self.hunger_timer = 0  # Reset after penalty

        return float(reward)

    def _calculate_collision_penalty_v3(self, prev_head: Optional[Tuple[int, int]], 
                                        snake_length: int) -> float:
        """Calculate gentler collision penalty for learning.
        
        Reduced penalties:
        - Wall collision: -10 (was -20)
        - Body collision (far): -10 (was -10 to -20)
        - Body collision (near head): -20 to -30 (was -30 to -50)
        
        Earlier in game = gentler penalty (more room for exploration)
        """
        if prev_head is None:
            return -10.0
        
        head_row, head_col = prev_head
        direction = self.game.direction
        
        # Check for wall collision
        if direction == "UP" and head_row == 0:
            return -10.0
        elif direction == "DOWN" and head_row == self.board_size - 1:
            return -10.0
        elif direction == "LEFT" and head_col == 0:
            return -10.0
        elif direction == "RIGHT" and head_col == self.board_size - 1:
            return -10.0
        
        # Body collision - find segment
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
            return -10.0
        
        try:
            segment_index = self.game.snake.index(collision_pos)
            
            # Gentler progressive penalty
            if segment_index == 1:  # Neck
                return -30.0  # Was -50
            elif segment_index <= 3:  # Near head
                return -20.0  # Was -30
            else:  # Far from head
                return -10.0  # Was -10 to -20
        except (ValueError, AttributeError):
            return -10.0

    def _is_on_edge(self) -> bool:
        """Check if snake head is on the board edge."""
        if not self.game.snake:
            return False
        
        head_row, head_col = self.game.snake[0]
        return (head_row == 0 or head_row == self.board_size - 1 or
                head_col == 0 or head_col == self.board_size - 1)
    
    def _is_wall_collision(self, prev_head: Optional[Tuple[int, int]]) -> bool:
        """Check if the collision was with a wall."""
        if prev_head is None:
            return True
        
        head_row, head_col = prev_head
        direction = self.game.direction
        
        # Check for wall collision
        if direction == "UP" and head_row == 0:
            return True
        elif direction == "DOWN" and head_row == self.board_size - 1:
            return True
        elif direction == "LEFT" and head_col == 0:
            return True
        elif direction == "RIGHT" and head_col == self.board_size - 1:
            return True
        
        return False

    def _is_near_body(self) -> bool:
        """Check if head is adjacent to body (excluding neck)."""
        if not self.game.snake or len(self.game.snake) < 4:
            return False
        
        head_row, head_col = self.game.snake[0]
        body_segments = self.game.snake[2:]
        
        adjacent_positions = [
            (head_row - 1, head_col),
            (head_row + 1, head_col),
            (head_row, head_col - 1),
            (head_row, head_col + 1),
        ]
        
        for pos in adjacent_positions:
            if pos in body_segments:
                return True
        
        return False

    def _is_trapped(self) -> bool:
        """Check if no safe moves available."""
        if not self.game.snake:
            return False
        
        head_row, head_col = self.game.snake[0]
        safe_moves = 0
        
        if head_row > 0 and (head_row - 1, head_col) not in self.game.snake:
            safe_moves += 1
        if head_row < self.board_size - 1 and (head_row + 1, head_col) not in self.game.snake:
            safe_moves += 1
        if head_col > 0 and (head_row, head_col - 1) not in self.game.snake:
            safe_moves += 1
        if head_col < self.board_size - 1 and (head_row, head_col + 1) not in self.game.snake:
            safe_moves += 1
        
        return safe_moves == 0

    def _is_center_crowded(self) -> bool:
        """Check if center area is crowded with snake body.
        
        Center is defined as the middle 50% of the board.
        """
        if not self.game.snake:
            return False
        
        # Define center boundaries
        margin = self.board_size // 4
        center_min = margin
        center_max = self.board_size - margin
        
        # Count body segments in center
        center_segments = 0
        for row, col in self.game.snake:
            if center_min <= row < center_max and center_min <= col < center_max:
                center_segments += 1
        
        # Consider crowded if more than 50% of snake is in center
        return center_segments > len(self.game.snake) * 0.5

    def _would_have_collided(self, prev_direction: str) -> bool:
        """Check if continuing in previous direction would have caused collision.
        
        This helps identify smart defensive turns.
        """
        if not self.game.snake or not prev_direction:
            return False
        
        head_row, head_col = self.game.snake[0]
        
        # Calculate where we would be if we continued
        if prev_direction == "UP":
            next_pos = (head_row - 1, head_col)
        elif prev_direction == "DOWN":
            next_pos = (head_row + 1, head_col)
        elif prev_direction == "LEFT":
            next_pos = (head_row, head_col - 1)
        elif prev_direction == "RIGHT":
            next_pos = (head_row, head_col + 1)
        else:
            return False
        
        # Check if that position would be a collision
        next_row, next_col = next_pos
        
        # Wall collision
        if (next_row < 0 or next_row >= self.board_size or 
            next_col < 0 or next_col >= self.board_size):
            return True
        
        # Body collision (need to check snake at PREVIOUS timestep, not current)
        # Since we already moved, self.game.snake is current state
        # We need to check if next_pos would have hit previous snake body
        # This is approximate - check if it's in current body (excluding new head)
        if next_pos in self.game.snake[1:]:
            return True
        
        return False

    def _manhattan_distance(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> float:
        """Calculate Manhattan distance between two positions."""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def _get_info(self) -> dict:
        """Get additional info about the environment state."""
        return {
            'score': self.game.score,
            'snake_length': len(self.game.snake) if self.game.snake else 0,
            'episode_length': self.episode_length,
            'turn_count': self.turn_count,
            'successful_turns': self.successful_turns,
            'edge_time': self.edge_time,
            'edge_ratio': self.edge_time / max(1, self.episode_length),
            'is_trapped': self._is_trapped() if self.game.snake else False,
            'difficulty_level': self.difficulty_level,
            'curriculum_stage': self.curriculum_stage,
            'steps_since_food': self.steps_since_food,
            'is_hungry': self.steps_since_food >= self.hunger_threshold if self.curriculum_stage == "aggressive" else False,
        }


if __name__ == "__main__":
    # Quick test of the curriculum-optimized environment
    print("Testing GymSnakeEnvV3 (Curriculum-Optimized)...")
    env = GymSnakeEnvV3(board_size=6, render_mode=None, seed=42)
    
    obs, info = env.reset()
    print(f"Initial observation shape: {obs.shape}")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    print(f"Difficulty level: {info['difficulty_level']}")
    print(f"Initial info: {info}")
    
    total_reward = 0
    food_count = 0
    
    print("\nRunning 5 test episodes...")
    for episode in range(5):
        obs, info = env.reset()
        episode_reward = 0
        episode_food = 0
        
        for step in range(200):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            
            if info.get('food_obtained', False):
                episode_food += 1
            
            if terminated or truncated:
                break
        
        total_reward += episode_reward
        food_count += episode_food
        
        print(f"Episode {episode+1}: Score={info['score']}, "
              f"Length={info['snake_length']}, "
              f"Reward={episode_reward:.1f}, "
              f"Edge%={info['edge_ratio']*100:.1f}%, "
              f"Turns={info['successful_turns']}/{info['turn_count']}")
    
    print(f"\nAverage reward: {total_reward/5:.1f}")
    print(f"Average food: {food_count/5:.1f}")
    print("\n✓ Curriculum environment test completed!")
