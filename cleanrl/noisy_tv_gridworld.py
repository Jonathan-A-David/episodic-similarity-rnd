import gymnasium as gym
from gymnasium import spaces
import numpy as np

class NoisyTVGridWorld(gym.Env):
    def __init__(self):
        super().__init__()
        self.grid_size = 7
        self.action_space = spaces.Discrete(4)
        # Observation space: [x_normalized, y_normalized, noisy_tv_pixel]
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, -10.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 10.0], dtype=np.float32),
            dtype=np.float32
        )
        self.x = 0
        self.y = 0
        self.steps_taken = 0
        self.max_steps = 100

    def _get_obs(self):
        # Trap Room at (6, 0)
        if self.x == 6 and self.y == 0:
            tv_noise = float(np.random.normal())
        else:
            tv_noise = 0.0
        return np.array([self.x / 6.0, self.y / 6.0, tv_noise], dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.x = 0
        self.y = 0
        self.steps_taken = 0
        return self._get_obs(), {}

    def step(self, action):
        self.steps_taken += 1
        
        # 0: Up, 1: Down, 2: Left, 3: Right
        if action == 0:
            self.y = min(self.y + 1, 6)
        elif action == 1:
            self.y = max(self.y - 1, 0)
        elif action == 2:
            self.x = max(self.x - 1, 0)
        elif action == 3:
            self.x = min(self.x + 1, 6)

        obs = self._get_obs()
        
        # Sparse Reward at Goal (6, 6)
        terminated = False
        reward = 0.0
        if self.x == 6 and self.y == 6:
            terminated = True
            reward = 1.0
            
        truncated = False
        if self.steps_taken >= self.max_steps:
            truncated = True

        info = {
            "x": self.x,
            "y": self.y
        }
        return obs, reward, terminated, truncated, info
