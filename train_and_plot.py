import numpy as np
import gym
import matplotlib.pyplot as plt
import os
import argparse
from collections import defaultdict

SEEDS = [0, 1, 2, 3, 4]
EPISODES = 200

class Agent:
    def __init__(self, n_states, n_actions, cbow=False, use_belief=False):
        self.n_states = n_states
        self.n_actions = n_actions
        self.Q = np.zeros((n_states, n_actions))
        self.visit_counts = np.ones((n_states, n_actions))
        self.use_belief = use_belief
        self.cbow = cbow
        self.embedding = self._init_embedding()

    def _init_embedding(self):
        d = 16  # Embedding dimension
        embedding = {
            's': np.random.randn(self.n_states, d),
            'a': np.random.randn(self.n_actions, d)
        }
        return embedding

    def act(self, state, epsilon):
        if np.random.rand() < epsilon:
            return np.random.choice(self.n_actions)
        if self.use_belief:
            return self._belief_based_action(state)
        return np.argmax(self.Q[state])

    def _belief_based_action(self, state):
        state_vec = self.embedding['s'][state]
        scores = np.zeros(self.n_actions)
        for a in range(self.n_actions):
            action_vec = self.embedding['a'][a]
            dist = np.linalg.norm(state_vec - action_vec)
            scores[a] = -dist  # Lower distance → higher score
        return np.argmax(scores)

    def update(self, s, a, r, s_next, alpha=0.1, gamma=0.99):
        td_target = r + gamma * np.max(self.Q[s_next])
        td_error = td_target - self.Q[s, a]
        self.Q[s, a] += alpha * td_error
        self.visit_counts[s, a] += 1

        if self.use_belief:
            self._update_embeddings(s, a)

    def _update_embeddings(self, s, a):
        d = self.embedding['s'].shape[1]
        s_vec = self.embedding['s'][s]
        a_vec = self.embedding['a'][a]

        # Gradient-like update to bring s and a closer
        delta = s_vec - a_vec
        self.embedding['s'][s] -= 0.01 * delta
        self.embedding['a'][a] += 0.01 * delta


def train(seed, cbow_flag=False, use_belief=False):
    env = gym.make("CartPole-v0")
    env.reset(seed=seed)
    np.random.seed(seed)

    n_states = env.observation_space.shape[0]
    n_actions = env.action_space.n
    agent = Agent(n_states=500, n_actions=n_actions, cbow=cbow_flag, use_belief=use_belief)

    rewards = []
    for ep in range(EPISODES):
        s, _ = env.reset()
        s = hash(tuple(s)) % 500
        done = False
        total_reward = 0
        epsilon = max(0.1, 1 - ep / 150)

        while not done:
            a = agent.act(s, epsilon)
            s_next, r, term, trunc, _ = env.step(a)
            done = term or trunc
            s_next = hash(tuple(s_next)) % 500
            agent.update(s, a, r, s_next)
            s = s_next
            total_reward += r

        rewards.append(total_reward)
    env.close()
    return np.array(rewards)


def save_reward_curves(all_curves, env_name):
    save_dir = f"results/{env_name}"
    os.makedirs(save_dir, exist_ok=True)

    colors = {"dqn": "red", "cbow": "green", "skipgram": "blue"}
    plt.figure()

    for method, curves in all_curves.items():
        mean = np.mean(curves, axis=0)
        std = np.std(curves, axis=0)
        x = range(len(mean))

        plt.plot(x, mean, label=method.upper(), color=colors[method])
        plt.fill_between(x, np.clip(mean - std, 0, 200), np.clip(mean + std, 0, 200), alpha=0.3, color=colors[method])

        # Save individual reward .npy
        np.save(os.path.join(save_dir, f"reward_{method}.npy"), np.array(curves))

    plt.title("Reward Curve over Training")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, "reward_curves.png"))
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="CartPole-v0", help="Name of the Gymnasium environment")
    args = parser.parse_args()
    env_name = args.env

    methods = {
        "dqn": {"cbow_flag": False, "use_belief": False},
        "cbow": {"cbow_flag": True, "use_belief": True},
        "skipgram": {"cbow_flag": False, "use_belief": True},
    }

    all_curves = {}
    for method, flags in methods.items():
        print(f"Training {method.upper()} on {env_name}...")
        curves = [train(seed, flags["cbow_flag"], flags["use_belief"]) for seed in SEEDS]
        all_curves[method] = curves

    save_reward_curves(all_curves, env_name)

