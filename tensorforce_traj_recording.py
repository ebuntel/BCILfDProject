from tokenize import String
import pyvirtualdisplay
import gym
import pickle
import os
import argparse

from datetime import datetime
from itertools import count
from tensorforce import Environment, Agent
from gym.utils import play
from tensorforce_pacman_env import tf_pacman_env

import ale_py.roms as roms
import tensorflow as tf
import tf_agents as tf_a
import matplotlib.pyplot as plt
import numpy as np


def main():
    #display = pyvirtualdisplay.Display(visible=1, size=(1400, 900)).start()

    arg_parser = argparse.ArgumentParser()

    arg_parser.add_argument('--convert_trajectory', metavar='c', type=bool, default=False,
            help="a bool indicating whether to convert a trajectory to tensorforce format or not")

    arg_parser.add_argument('--trajectory_in', metavar='i', type=str,
            help="the path to a trajectory for converting")

    arg_parser.add_argument('--pretrain', metavar='p', type=bool, default=False,
            help="a bool indicating whether to pretrain a Tensorforce agent or not")

    args = arg_parser.parse_args()

    write_traj = args.convert_trajectory

    write_in = args.trajectory_in

    pretrain = args.pretrain

    if(write_traj != False):
        env_name = 'ALE/MsPacman-v5'

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

        write_custom_recording_file(in_directory=write_in, out_directory='pacman_traces', env=env)

    if(pretrain):

        # Counts # of trace files in folder
        _, _, files = next(os.walk("./pacman_traces"))
        trace_count = len(files)

        environment = Environment.create(environment=tf_pacman_env)
        print(environment.reset())
        agent = Agent.create(agent='dqn', environment=environment, batch_size = 1, memory = 10000) 
        print("Agent Created")
        agent.pretrain(directory='pacman_traces', num_iterations=30, num_traces=trace_count, num_updates=1)
        print("Agent pretrained")

        # runner = Runner(agent=agent, environment=environment)
        # runner.run(num_episodes=100, evaluation=True)
        # runner.close()

        # # Close agent and environment
        agent.close()
        environment.close()

def write_custom_recording_file(in_directory, out_directory, env):
    environment = Environment.create(environment=env)

    with open(in_directory, "rb") as trajectoryfile:
        trajectories = pickle.load(trajectoryfile)
    
    actions = []

    for i in range(len(trajectories)):
        print(trajectories[i].action)
        actions.append(trajectories[i][0])

    #print(actions)

    # Record episode experience
    episode_states = list()
    episode_actions = list()
    episode_terminal = list()
    episode_reward = list()

    states = environment.reset()
    terminal = False
    while not terminal and len(actions) >= 1:
        episode_states.append(states)
        action = actions.pop(0)
        episode_actions.append(action)
        states, terminal, reward = environment.execute(actions=action)
        if(len(actions) == 0):
            terminal = True
        episode_terminal.append(terminal)
        episode_reward.append(reward)

    print("# of states: ", len(episode_states))
    print("# of actions: ", len(episode_actions))
    print("# of terminal bools: ", len(episode_terminal))
    print("# of rewards: ", len(episode_reward))

    # Write recorded episode trace to npz file

    # Counts # of trace files
    _, _, files = next(os.walk("./pacman_traces"))
    file_count = len(files)

    #num = len([name for name in os.listdir('~/Research/RL/BCILfDProject/pacman_traces') if os.path.isfile(name)])

    #print(num)

    np.savez_compressed(
        file=os.path.join(out_directory, 'trace-{:09d}.npz'.format(file_count + 1)),
        states=np.stack(episode_states, axis=0),
        actions=np.stack(episode_actions, axis=0),
        terminal=np.stack(episode_terminal, axis=0),
        reward=np.stack(episode_reward, axis=0)
    )

if __name__ ==  '__main__':
    main()