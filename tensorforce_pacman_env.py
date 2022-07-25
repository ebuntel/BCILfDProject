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
        env._max_episode_steps = None 

        self.env = env
        self.obs = np.array(self.env.reset())
        self._expect_receive = None
        self.reward_list = []
        self.episode_reward_list = []
        super().__init__()

    def max_episode_timesteps(self):
      return self.env._max_episode_steps

    def set_max_episode_timesteps(self, stepnum):
        self.env._max_episode_steps = stepnum

    # Required for Tensorforce
    def states(self):
        return dict(type='float', shape=(4,84,84))

    def actions(self):
        return dict(type='int', num_values=18)

    def reset(self):
        self.obs = np.array(self.env.reset())

        rew_sum = np.sum(self.reward_list)

        if(rew_sum != 0):
            self.episode_reward_list.append(rew_sum)
        self.reward_list = []
        return np.array(self.obs)

    def execute(self, actions):
        new_obs, reward, terminal, __ = self.env.step(actions)

        self.reward_list.append(reward)

        self.obs = np.array(new_obs)
        self._expect_receive = None
        self._actions = None
        return new_obs, terminal, reward
