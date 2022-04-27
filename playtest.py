
from itertools import count
import matplotlib.pyplot as plt
import numpy as np
import pyvirtualdisplay
import gym
import pickle

from gym import wrappers
from datetime import datetime

import ale_py.roms as roms

import tensorflow as tf
from functools import partial

from tf_agents.environments import suite_gym
from tf_agents.environments import tf_py_environment

from tf_agents.environments.atari_preprocessing import AtariPreprocessing
from tf_agents.environments.atari_wrappers import FrameStack4

# Set up a virtual display for rendering OpenAI gym environments.
display = pyvirtualdisplay.Display(visible=1, size=(1400, 900)).start()

num_iterations = 200000 # @param {type:"integer"}

initial_collect_steps = 100  # @param {type:"integer"}
collect_steps_per_iteration =   1# @param {type:"integer"}
replay_buffer_max_length = 100000  # @param {type:"integer"}

batch_size = 64  # @param {type:"integer"}
learning_rate = 25e-5  # @param {type:"number"}
log_interval = 200  # @param {type:"integer"}

num_eval_episodes = 10  # @param {type:"integer"}
eval_interval = 1000  # @param {type:"integer"}

step_record = []

env_name = 'ALE/MsPacman-v5'

def save_to_disk(steps):
    try:
        now = datetime.now()
        date_time = now.strftime("%m-%d-%Y,%H-%M")
        name_string = "trajectory" + date_time + ".pkl"
        print(name_string)
        with open(name_string,'wb') as outputloc:
            pickle.dump(steps, outputloc, pickle.HIGHEST_PROTOCOL)
        step_record = []
    except:
        print("Error writing trajectory to disk.")

def callback(obs_t, obs_tp1, action, rew, done, info):
    step_record.append((obs_t, obs_tp1, action, rew, done, info))
    if(done):
        save_to_disk(step_record)
    else:
        return [rew, obs_t, action]

#Atari preprocessing and Frame stacking wrappers
# gym_env_wrappers = [partial(
#                         #  gym.wrappers.AtariPreprocessing,
#                          AtariPreprocessing,
#                          noop_max=30, 
#                          frame_skip=4, 
#                          screen_size=84, 
#                          terminal_on_life_loss=False, 
#                          grayscale_obs=True, 
#                          grayscale_newaxis=False, 
#                          scale_obs=False,
#                          ),
#                      partial(
#                          gym.wrappers.FrameStack, 
#                          num_stack=4)
#                    ]

#initial environment
#env = suite_gym.load(env_name, 
#                     gym_env_wrappers=[AtariPreprocessing, FrameStack4])

env = gym.make(env_name)

from gym.utils import play
env.reset()
adict = env.unwrapped.get_keys_to_action()
print(adict)

train_py_env = suite_gym.load(env_name, 
                     gym_env_wrappers=[AtariPreprocessing, FrameStack4])
eval_py_env = suite_gym.load(env_name, 
                     gym_env_wrappers=[AtariPreprocessing, FrameStack4])

train_env = tf_py_environment.TFPyEnvironment(train_py_env)
eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)

play.play(env, keys_to_action = adict, zoom = 4, fps = 15, callback = callback)

print(len(step_record))
print(step_record[0])
