import gymnasium as gym
from gymnasium import spaces
import numpy as np

class NoisyTVKeyDoorGridWorld(gym.Env):
    def __init__(self, grid_size=11, noise_scale=1.0):
        super().__init__()
        self.grid_size = grid_size
        self.noise_scale = noise_scale
        
        # Calculate dynamic game coordinates
        self.tv_x = grid_size // 2 - 1
        self.tv_y = 0
        
        self.wall_x = grid_size // 2
        self.door_x = grid_size // 2
        self.door_y = grid_size // 2
        
        self.key_x = 0
        self.key_y = grid_size - 1
        
        self.goal_x = grid_size - 1
        self.goal_y = grid_size - 1
        
        self.action_space = spaces.Discrete(4)
        # Observation space: [x_normalized, y_normalized, has_key, door_unlocked, tv_noise]
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0, 0.0, -10.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0, 1.0, 10.0], dtype=np.float32),
            dtype=np.float32
        )
        self.x = 0
        self.y = 0
        self.has_key = 0.0
        self.door_unlocked = 0.0
        self.steps_taken = 0
        self.max_steps = 200

    def _get_obs(self):
        # Noisy TV at (tv_x, tv_y)
        if self.x == self.tv_x and self.y == self.tv_y:
            tv_noise = float(np.random.normal(scale=self.noise_scale))
        else:
            tv_noise = 0.0
        return np.array([
            self.x / float(self.grid_size - 1),
            self.y / float(self.grid_size - 1),
            self.has_key,
            self.door_unlocked,
            tv_noise
        ], dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.x = 0
        self.y = 0
        self.has_key = 0.0
        self.door_unlocked = 0.0
        self.steps_taken = 0
        return self._get_obs(), {}

    def step(self, action):
        self.steps_taken += 1
        
        # 0: Up, 1: Down, 2: Left, 3: Right
        nx, ny = self.x, self.y
        if action == 0:
            ny = min(self.y + 1, self.grid_size - 1)
        elif action == 1:
            ny = max(self.y - 1, 0)
        elif action == 2:
            nx = max(self.x - 1, 0)
        elif action == 3:
            nx = min(self.x + 1, self.grid_size - 1)

        # Wall and Door Mechanics
        move_allowed = True
        if nx == self.wall_x: # Wall Column
            if ny != self.door_y: # Non-door wall segment
                move_allowed = False
            else: # Door cell
                if self.door_unlocked == 0.0:
                    if self.has_key == 1.0:
                        self.door_unlocked = 1.0
                        self.has_key = 0.0  # Key consumed to unlock
                    else:
                        move_allowed = False  # Locked!

        if move_allowed:
            self.x, self.y = nx, ny

        # Key Pickup Check: (key_x, key_y)
        if self.x == self.key_x and self.y == self.key_y and self.has_key == 0.0 and self.door_unlocked == 0.0:
            self.has_key = 1.0

        obs = self._get_obs()
        
        # Sparse Goal at (goal_x, goal_y) - requires door to be unlocked
        terminated = False
        reward = 0.0
        if self.x == self.goal_x and self.y == self.goal_y and self.door_unlocked == 1.0:
            terminated = True
            reward = 1.0
            
        truncated = False
        if self.steps_taken >= self.max_steps:
            truncated = True

        info = {
            "x": self.x,
            "y": self.y,
            "has_key": self.has_key,
            "door_unlocked": self.door_unlocked
        }
        return obs, reward, terminated, truncated, info
