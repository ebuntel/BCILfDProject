import gym

import ale_py.roms as roms

from tensorforce import Environment

class tf_pacman_env(gym.Env, Environment):
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
        super().__init__()

    # Required
    def states(self):
        return super().states()
