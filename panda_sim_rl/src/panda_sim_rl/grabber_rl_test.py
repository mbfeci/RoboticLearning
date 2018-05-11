import numpy as np
from gym import spaces

class RlTest:
    def __init__(self):
        action_high = np.array([1., 1., 1.])
        high = np.array([10., 10., 10., 1., 1., 1.])

        self.action_space = spaces.Box(low=-action_high, high=action_high)
        self.observation_space = spaces.Box(low=-high, high=high)
        self.reset()

    def reset(self):
        self.gripper_pos = np.random.uniform(-1, 1, 3)
        self.dest_pos = np.random.uniform(-1, 1, 3)
        return np.concatenate((self.gripper_pos, self.dest_pos))
    
    def step(self, action):
        action = np.clip(action, -1, 1)
        prev_dist = self.get_distance()
        state = self.update_state(action)
        reward, done = self.get_reward_done(prev_dist)
        return state, reward, done, None

    def get_distance(self):
        return np.linalg.norm(self.dest_pos - self.gripper_pos)

    def get_reward_done(self, prev_dist):
        done = False    
        current_dist = self.get_distance()
        reward = prev_dist - current_dist - 1
        if current_dist <= 0.1:
            reward += 100
            done = True
        return reward, done

    def update_state(self, action):
        self.gripper_pos = self.gripper_pos + action
        return np.concatenate((self.gripper_pos, self.dest_pos))

    def render(self):
        return None