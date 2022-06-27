import gym

import ale_py.roms as roms

class tf_pacman_env(gym.Env):
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
