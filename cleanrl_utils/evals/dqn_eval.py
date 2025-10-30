# dqn_eval.py
# Adapted from CleanRL eval utils
import random
from typing import Callable
import numpy as np
import torch
import gymnasium as gym


def evaluate(
    model_path: str = None,
    make_env=None,
    env_id="CartPole-v1",
    eval_episodes=5,
    run_name="eval",
    Model=None,
    device=torch.device("cpu"),
    epsilon=0.05,
    capture_video=False,
    model_instance=None
):
    envs = gym.vector.SyncVectorEnv([make_env(env_id, 0, 0, capture_video, run_name)])

    if model_instance is not None:
        model = model_instance
    else:
        model = Model(envs).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))

    model.eval()
    obs, _ = envs.reset()
    episodic_returns = []
    episode_return = np.zeros(envs.num_envs, dtype=np.float32)

    while len(episodic_returns) < eval_episodes:
        if random.random() < epsilon:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            q_values = model(torch.tensor(obs, dtype=torch.float32, device=device))
            actions = torch.argmax(q_values, dim=1).cpu().numpy()

        next_obs, rewards, terminations, truncations, infos = envs.step(actions)
        episode_return += rewards
        done_flags = np.logical_or(terminations, truncations)
        for i, done in enumerate(done_flags):
            if done:
                episodic_returns.append(episode_return[i])
                episode_return[i] = 0.0
        obs = next_obs

    envs.close()
    return episodic_returns
