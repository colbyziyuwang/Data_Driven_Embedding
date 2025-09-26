import argparse
import gymnasium as gym  # ← switched to Gymnasium
import numpy as np
import torch
import tqdm
import matplotlib.pyplot as plt
import os

from torch.utils.data import DataLoader
from embedding_cbow import StateActionPredictionModel
from embedding_skipgram import SkipGramActionPredictionModel
from load_data import StateActionDataset, SkipGramStateActionDataset
import utils.envs, utils.seed, utils.buffers, utils.torch

# Global config
SEEDS = [1, 2, 3, 4, 5]
t = utils.torch.TorchHelper()
DEVICE = t.device

# Hyperparameters
OBS_N = 4
ACT_N = 2
MINIBATCH_SIZE = 10
GAMMA = 0.99
LEARNING_RATE = 5e-4
TRAIN_AFTER_EPISODES = 10
TRAIN_EPOCHS = 5
BUFSIZE = 10000
EPISODES = 300
TEST_EPISODES = 1
HIDDEN = 512
TARGET_UPDATE_FREQ = 10
STARTING_EPSILON = 1.0
STEPS_MAX = 10000
EPSILON_END = 0.01

EPSILON = STARTING_EPSILON
Q = None

def create_everything(seed, cbow_flag, env_name):
    utils.seed.seed(seed)
    env = gym.make(env_name)
    env.reset(seed=seed)
    test_env = gym.make(env_name)
    test_env.reset(seed=10 + seed)

    buf = utils.buffers.ReplayBuffer(BUFSIZE)
    Q = torch.nn.Sequential(
        torch.nn.Linear(OBS_N, HIDDEN), torch.nn.ReLU(),
        torch.nn.Linear(HIDDEN, HIDDEN), torch.nn.ReLU(),
        torch.nn.Linear(HIDDEN, ACT_N)
    ).to(DEVICE)

    Qt = torch.nn.Sequential(
        torch.nn.Linear(OBS_N, HIDDEN), torch.nn.ReLU(),
        torch.nn.Linear(HIDDEN, HIDDEN), torch.nn.ReLU(),
        torch.nn.Linear(HIDDEN, ACT_N)
    ).to(DEVICE)

    OPT = torch.optim.Adam(Q.parameters(), lr=LEARNING_RATE)
    
    embed_model = StateActionPredictionModel(OBS_N, ACT_N) if cbow_flag else SkipGramActionPredictionModel(OBS_N, ACT_N)
    return env, test_env, buf, Q, Qt, OPT, embed_model

def update(target, source):
    for tp, p in zip(target.parameters(), source.parameters()):
        tp.data.copy_(p.data)

def policy(obs):
    global EPSILON, Q
    obs = t.f(obs[0] if isinstance(obs, tuple) else obs).view(-1, OBS_N)
    if np.random.rand() < EPSILON:
        action = np.random.randint(ACT_N)
    else:
        action = torch.argmax(Q(obs)).item()
    EPSILON = max(EPSILON_END, EPSILON - (1.0 / STEPS_MAX))
    return action

def update_networks(epi, buf, Q, Qt, OPT, embed_model, use_belief):
    S, A, R, S2, D = buf.sample(MINIBATCH_SIZE, t)
    qvalues = Q(S).gather(1, A.view(-1, 1)).squeeze()
    q2values_all = Qt(S2)

    if use_belief:
        weights = embed_model.compute_distance(S2)
        q2values = torch.sum(weights * q2values_all, dim=1)
    else:
        q2values = torch.max(q2values_all, dim=1).values

    targets = R + GAMMA * q2values * (1 - D)
    loss = torch.nn.MSELoss()(targets.detach(), qvalues)

    OPT.zero_grad()
    loss.backward()
    OPT.step()

    if epi % TARGET_UPDATE_FREQ == 0:
        update(Qt, Q)
    return loss.item()

def create_batch_data(state_action_sequences, cbow):
    data = []
    if cbow:
        for seq in state_action_sequences:
            for i in range(0, len(seq) - 6, 2):
                data.append((seq[i], seq[i+1], seq[i+2], seq[i+3], seq[i+4], seq[i+5]))
        dataset = StateActionDataset(data)
    else:
        for seq in state_action_sequences:
            for i in range(1, len(seq) - 2, 2):
                data.append((seq[i], seq[i+1], seq[i+2]))
        dataset = SkipGramStateActionDataset(data)
    return DataLoader(dataset, batch_size=16, shuffle=True)

def train(seed, cbow_flag, use_belief, env_name):
    global EPSILON, Q
    print(f"Training seed {seed} ({'cbow' if cbow_flag else 'skipgram' if use_belief else 'dqn'})...\n")
    env, test_env, buf, Q, Qt, OPT, embed_model = create_everything(seed, cbow_flag, env_name)
    EPSILON = STARTING_EPSILON
    test_rewards = []
    state_action_sequences = []
    pbar = tqdm.trange(EPISODES)

    for epi in pbar:
        # Updated for gymnasium reset() and step() return signatures
        _, _, R, s_a_seq = utils.envs.play_episode_rb(env, policy, buf)
        state_action_sequences.append(s_a_seq)

        if epi >= TRAIN_AFTER_EPISODES:
            for _ in range(TRAIN_EPOCHS):
                data_loader = create_batch_data(state_action_sequences, cbow_flag)
                embed_model.train(data_loader)
                update_networks(epi, buf, Q, Qt, OPT, embed_model, use_belief)
            state_action_sequences = []

        test_r = [sum(utils.envs.play_episode(test_env, policy)[2]) for _ in range(TEST_EPISODES)]
        avg_r = np.mean(test_r)
        test_rewards.append(avg_r)
        pbar.set_description(f"AvgR25: {np.mean(test_rewards[-25:]):.2f}")

    env.close()
    return test_rewards

def plot_combined(curves_dict, fname):
    plt.figure()
    for method, data in curves_dict.items():
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        x = range(len(mean))
        color = {'dqn': 'blue', 'cbow': 'green', 'skipgram': 'red'}[method]
        plt.plot(x, mean, label=method.upper(), color=color)
        plt.fill_between(x, np.clip(mean - std, 0, 200), np.clip(mean + std, 0, 200), alpha=0.3, color=color)

    plt.title("Reward Curve over Training")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.legend()
    plt.grid(True)
    plt.savefig(fname)
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="CartPole-v0", help="Gym environment name")
    args = parser.parse_args()
    env_name = args.env

    env = gym.make(env_name)
    obs_space = env.observation_space
    act_space = env.action_space
    OBS_N = obs_space.shape[0]
    ACT_N = act_space.n

    os.makedirs(f"results/{env_name}", exist_ok=True)

    curves_dict = {}

    for method in ["dqn", "cbow", "skipgram"]:
        cbow_flag = method == "cbow"
        use_belief = method in ["cbow", "skipgram"]
        curves = [train(seed, cbow_flag, use_belief, env_name) for seed in SEEDS]
        curves_dict[method] = np.array(curves)
        np.save(f"results/{env_name}/{method}.npy", curves_dict[method])

    plot_combined(curves_dict, f"results/{env_name}/reward_curves.png")
