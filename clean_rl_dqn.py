# Adapted from CleanRL: https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/dqn.py

import os
import random
import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from torch.utils.tensorboard import SummaryWriter

from cleanrl_utils.buffers import ReplayBuffer
from cleanrl_utils.evals.dqn_eval import evaluate

from embedding_cbow import StateActionPredictionModel
from embedding_skipgram import SkipGramActionPredictionModel
from torch.utils.data import DataLoader, TensorDataset
from load_data import StateActionDataset, SkipGramStateActionDataset

@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    save_model: bool = False
    """whether to save model into the `runs/{run_name}` folder"""
    upload_model: bool = False
    """whether to upload the saved model to huggingface"""
    hf_entity: str = ""
    """the user or org name of the model repository from the Hugging Face Hub"""

    # Algorithm specific arguments
    env_id: str = "CartPole-v1"
    """the id of the environment"""
    total_timesteps: int = 500000
    """total timesteps of the experiments"""
    learning_rate: float = 2.5e-4
    """the learning rate of the optimizer"""
    num_envs: int = 1
    """the number of parallel game environments"""
    buffer_size: int = 10000
    """the replay memory buffer size"""
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 1.0
    """the target network update rate"""
    target_network_frequency: int = 500
    """the timesteps it takes to update the target network"""
    batch_size: int = 128
    """the batch size of sample from the reply memory"""
    start_e: float = 1
    """the starting epsilon for exploration"""
    end_e: float = 0.05
    """the ending epsilon for exploration"""
    exploration_fraction: float = 0.5
    """the fraction of `total-timesteps` it takes from start-e to go end-e"""
    learning_starts: int = 10000
    """timestep to start learning"""
    train_frequency: int = 10
    """the frequency of training"""

    algo_name: str = "DQN"
    """the name of the algorithm (DQN, CBOW, SkipGram)"""

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

def make_env(env_id, seed, idx, capture_video, run_name):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed)

        return env

    return thunk


# ALGO LOGIC: initialize agent here:
class QNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(np.array(env.single_observation_space.shape).prod(), 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, env.single_action_space.n),
        )

    def forward(self, x):
        return self.network(x)


def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)


if __name__ == "__main__":
    args = tyro.cli(Args)
    assert args.num_envs == 1, "vectorized envs are not supported at the moment"
    run_name = f"{args.env_id}__{args.algo_name}__seed{args.seed}__{int(time.time())}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, args.seed + i, i, args.capture_video, run_name) for i in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    q_network = QNetwork(envs).to(device)
    optimizer = optim.Adam(q_network.parameters(), lr=args.learning_rate)
    target_network = QNetwork(envs).to(device)
    target_network.load_state_dict(q_network.state_dict())

    # Set up CBOW and SkipGram Networks if needed
    obs_space = envs.single_observation_space
    act_space = envs.single_action_space
    obs_dim = obs_space.shape[0]
    act_dim = act_space.n

    embed_model = None
    if args.algo_name == "CBOW":
        embed_model = StateActionPredictionModel(obs_dim, act_dim, device)
    elif args.algo_name == "SkipGram":
        embed_model = SkipGramActionPredictionModel(obs_dim, act_dim, device)

    state_action_sequences = []

    rb = ReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device,
        handle_timeout_termination=False,
    )
    start_time = time.time()

    eval_returns = []  # to store the average reward over time
    eval_interval = 5000  # evaluate every 5k environment steps

    # TRY NOT TO MODIFY: start the game
    obs, _ = envs.reset(seed=args.seed)
    for global_step in range(args.total_timesteps):
        # ALGO LOGIC: put action logic here
        epsilon = linear_schedule(args.start_e, args.end_e, args.exploration_fraction * args.total_timesteps, global_step)
        if random.random() < epsilon:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            q_values = q_network(torch.Tensor(obs).to(device))
            actions = torch.argmax(q_values, dim=1).cpu().numpy()

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)

        for env_idx in range(envs.num_envs):
            seq = [obs[env_idx], actions[env_idx], next_obs[env_idx]]
            state_action_sequences.append(seq)
        state_action_sequences.append(seq)

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        if "final_info" in infos:
            for info in infos["final_info"]:
                if info and "episode" in info:
                    print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                    writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                    writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)

        # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
        real_next_obs = next_obs.copy()
        final_obs = infos.get("final_observation", None)

        if final_obs is not None:
            for idx, trunc in enumerate(truncations):
                if trunc and final_obs[idx] is not None:
                    real_next_obs[idx] = final_obs[idx]

        rb.add(obs, real_next_obs, actions, rewards, terminations, infos)

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        # ALGO LOGIC: training.
        if global_step > args.learning_starts:
            if global_step % args.train_frequency == 0:
                data = rb.sample(args.batch_size)

                # Update CBOW and Skipgram models if needed
                if embed_model is not None:
                    data_loader = create_batch_data(state_action_sequences, args.algo_name == "CBOW")
                    embed_model.train(data_loader)

                with torch.no_grad():
                    target_max = None
                    if (args.algo_name == "DQN"):
                        target_max, _ = target_network(data.next_observations).max(dim=1)
                    elif (args.algo_name in ["CBOW", "SkipGram"]):
                        weights = embed_model.compute_distance(data.next_observations)
                        q2values_all = target_network(data.next_observations)
                        target_max = torch.sum(weights * q2values_all, dim=1)

                    td_target = data.rewards.flatten() + args.gamma * target_max * (1 - data.dones.flatten())
                old_val = q_network(data.observations).gather(1, data.actions).squeeze()
                loss = F.mse_loss(td_target, old_val)

                if global_step % 100 == 0:
                    writer.add_scalar("losses/td_loss", loss, global_step)
                    writer.add_scalar("losses/q_values", old_val.mean().item(), global_step)
                    # print("SPS:", int(global_step / (time.time() - start_time)))
                    writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

                # optimize the model
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # update target network
            if global_step % args.target_network_frequency == 0:
                for target_network_param, q_network_param in zip(target_network.parameters(), q_network.parameters()):
                    target_network_param.data.copy_(
                        args.tau * q_network_param.data + (1.0 - args.tau) * target_network_param.data
                    )
            
            if global_step % eval_interval == 0 and global_step > args.learning_starts:
                returns = evaluate(
                    model_path=None,  # directly use q_network without saving
                    make_env=make_env,
                    env_id=args.env_id,
                    eval_episodes=5,
                    run_name=f"{args.env_id}_eval_{global_step}",
                    Model=QNetwork,
                    device=device,
                    epsilon=args.end_e,
                    capture_video=False,
                    model_instance=q_network  # <-- we’ll add this parameter
                )
                avg_return = np.mean(returns)
                eval_returns.append(avg_return)
                # writer.add_scalar("eval/avg_return", avg_return, global_step)

    envs.close()
    writer.close()

    save_dir = f"results_cleanrl/{args.env_id}/{args.algo_name}"
    os.makedirs(save_dir, exist_ok=True)
    np.save(f"{save_dir}/eval_returns_seed{args.seed}.npy", np.array(eval_returns))
