import gym
import numpy as np

import ale_py.roms as roms

from tensorforce import Environment

class tf_pacman_env(Environment):
    def __init__(self):
        env = gym.make("ALE/MsPacman-v5")
        env = gym.wrappers.AtariPreprocessing(
            env,
            noop_max = 30, 
            frame_skip = 1, 
            screen_size = 84, 
            terminal_on_life_loss = False,
            grayscale_obs = True, 
            grayscale_newaxis = False, 
            scale_obs = False)
        env = gym.wrappers.FrameStack(env, num_stack = 4)
        
        self.env = env
        self.obs = np.array(self.env.reset())
        super().__init__()

    # Required for Tensorforce
    def states(self):
        return dict(type='float', shape=(4,84,84))

    def actions(self):
        return dict(type='int', num_values=18)

    def reset(self):
        return self.env.reset()

    def execute(self, actions):
        new_obs, reward, terminal, info = self.env.step(actions)
        self.obs = np.array(new_obs)
        return new_obs, reward, terminal, info