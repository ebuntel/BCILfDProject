import pyvirtualdisplay
import gym
import pickle
import os

from datetime import datetime
from itertools import count
from tensorforce import Environment
from gym.utils import play

import ale_py.roms as roms
import tensorflow as tf
import tf_agents as tf_a
import matplotlib.pyplot as plt
import numpy as np


def main():
    display = pyvirtualdisplay.Display(visible=1, size=(1400, 900)).start()

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

    write_custom_recording_file(in_directory='', out_directory='pacman_traces', env=env)

    # Pretrain a new agent on the recorded traces: for 30 iterations, feed the
    # experience of one episode to the agent and subsequently perform one update
    # environment = Environment.create(environment=env)
    # agent = Agent.create(agent='benchmarks/configs/ppo.json', environment=environment)
    # agent.pretrain(directory='ppo-traces', num_iterations=30, num_traces=1, num_updates=1)

    # # Evaluate the pretrained agent
    # runner = Runner(agent=agent, environment=environment)
    # runner.run(num_episodes=100, evaluation=True)
    # runner.close()

    # # Close agent and environment
    # agent.close()
    # environment.close()

def write_custom_recording_file(in_directory, out_directory, env):
    # Start recording traces after 80 episodes -- by then, the environment is solved
    environment = Environment.create(environment=env)

    with open(in_directory, "rb") as trajectoryfile:
        trajectories = pickle.load(trajectoryfile)
    
    actions = []

    for trajectory in trajectories:
        actions.append(trajectories[1].action)

    # Record 20 episodes
    for episode in range(20):

        # Record episode experience
        episode_states = list()
        episode_actions = list()
        episode_terminal = list()
        episode_reward = list()

        # Evaluation episode
        states = environment.reset()
        terminal = False
        while not terminal:
            episode_states.append(states)
            action = actions.pop(0)
            episode_actions.append(action)
            states, terminal, reward = environment.execute(actions=action)
            episode_terminal.append(terminal)
            episode_reward.append(reward)

        # Write recorded episode trace to npz file
        np.savez_compressed(
            file=os.path.join(out_directory, 'trace-{:09d}.npz'.format(episode)),
            states=np.stack(episode_states, axis=0),
            actions=np.stack(episode_actions, axis=0),
            terminal=np.stack(episode_terminal, axis=0),
            reward=np.stack(episode_reward, axis=0)
        )

if __name__ ==  '__main__':
    main()