import base64
import imageio
import IPython
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import PIL.Image
import pyvirtualdisplay
import reverb
import gym

from gym import wrappers

import ale_py.roms as roms

import tensorflow as tf
from functools import partial

from tf_agents.agents.dqn import dqn_agent
from tf_agents.drivers import py_driver
from tf_agents.environments import suite_gym
from tf_agents.environments import tf_py_environment

from tf_agents.environments import atari_wrappers
from tf_agents.environments import py_environment
from tf_agents.environments import suite_atari

from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics
from tf_agents.networks import sequential
from tf_agents.networks import q_network
from tf_agents.policies import py_tf_eager_policy
from tf_agents.policies import random_tf_policy
from tf_agents.policies import epsilon_greedy_policy
from tf_agents.replay_buffers import reverb_replay_buffer
from tf_agents.replay_buffers import reverb_utils
from tf_agents.trajectories import trajectory
from tf_agents.specs import tensor_spec
from tf_agents.utils import common
from tf_agents.train.utils import train_utils

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

env_name = 'ALE/MsPacman-v5'

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

play.play(env, keys_to_action = adict, zoom = 4)