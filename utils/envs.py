import gym
import numpy as np
import random
from copy import deepcopy

# Play an episode according to a given policy
# env: environment
# policy: function(env, state)
# render: whether to render the episode or not (default - False)
def play_episode(env, policy, render = False):
    states, actions, rewards = [], [], []
    states.append(env.reset())
    done = False
    if render: env.render()
    while not done:
        action = policy(env, states[-1])
        actions.append(action)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        if render: env.render()
        states.append(obs)
        rewards.append(reward)
    return states, actions, rewards

# Play an episode according to a given policy and add 
# to a replay buffer (also collects state_action_sequences)
# env: environment
# policy: function(env, state)
# state_action_sequence: collects (s1, a1, s2, a2, ... sn)
def play_episode_rb(env, policy, buf):
    states, actions, rewards, state_action_sequence = [], [], [], []
    state0 = env.reset()
    states.append(state0)
    state_action_sequence.append(state0)
    done = False
    while not done:
        action = policy(env, states[-1])
        actions.append(action)
        state_action_sequence.append(action)
        obs, reward, terminated, truncated, info = env.step(action)
        state_action_sequence.append(obs)
        done = terminated or truncated
        buf.add(states[-1], action, reward, obs, done)
        states.append(obs)
        rewards.append(reward)
    return states, actions, rewards, state_action_sequence
