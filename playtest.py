
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

import tf_agents as tf_a

# from tf_agents.environments import suite_gym
# from tf_agents.environments import tf_py_environment

# from tf_agents.environments.atari_preprocessing import AtariPreprocessing
# from tf_agents.environments.atari_wrappers import FrameStack4

# Set up a virtual display for rendering OpenAI gym environments.
display = pyvirtualdisplay.Display(visible=1, size=(1400, 900)).start()

step_record = []

env_name = 'ALE/MsPacman-v5'

def save_to_disk(steps):
    #try:
    now = datetime.now()
    date_time = now.strftime("%m-%d-%Y,%H-%M")
    name_string = "trajectory" + date_time + ".pkl"
    print(name_string)

    traj_list = []

    for i in range(len(steps)):
        if(i == len(steps) - 1):
            #traj_list.append(tf_a.trajectories.from_transition(step_record[i][0], step_record[i][1], None))
            pass
        else:
            traj_list.append(tf_a.trajectories.from_transition(step_record[i][0], step_record[i][1], step_record[i][2]))
    step_record.clear()
    
    with open(name_string,'wb') as outputloc:
        pickle.dump(traj_list, outputloc, pickle.HIGHEST_PROTOCOL)
    #except:
    #    print("Error writing trajectory to disk.")

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


from gym.utils import play
env.reset()
adict = env.unwrapped.get_keys_to_action()
print(adict)

def callback(obs_t, obs_tp1, action, rew, done, info):

    tf_rew = tf.Tensor(np.array(rew, dtype=np.float32), dtype=np.float32)
    tf_rew.name = "Reward"

    tf_

    if(info['episode_frame_number'] == 1):
        step_type = tf_a.trajectories.StepType.FIRST
    elif(done):
        step_type = tf_a.trajectories.StepType.LAST
    else:
        step_type = tf_a.trajectories.StepType.MID

    tf_timestep = tf_a.trajectories.TimeStep(step_type, rew, 1.0, obs_t)
    tf_polstep = tf_a.trajectories.PolicyStep(action = action)

    step_record.append([tf_timestep, tf_polstep, None])

    if(info['episode_frame_number'] != 1):
        step_record[-1][2] = tf_timestep

    if(done):
        save_to_disk(step_record)
    else:
        return [rew, obs_t, action]

# train_py_env = tf_a.environments.suite_gym.load(env_name, 
#                      gym_env_wrappers=[tf_a.AtariPreprocessing, tf_a.FrameStack4])
# eval_py_env = tf_a.environments.suite_gym.load(env_name, 
#                      gym_env_wrappers=[tf_a.AtariPreprocessing, tf_a.FrameStack4])

# train_env = tf_a.environments.tf_py_environment.TFPyEnvironment(train_py_env)
# eval_env = tf_a.environments.tf_py_environment.TFPyEnvironment(eval_py_env)

play.play(env, keys_to_action = adict, zoom = 4, fps = 15, callback = callback)

print(len(step_record))
print(step_record[0])


