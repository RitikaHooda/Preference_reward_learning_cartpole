import torch
from torch.distributions.categorical import Categorical
import numpy as np
import gymnasium as gym
from matplotlib import pyplot as plt
from utils import mlp


def generate_rollout(policy, env, rendering=False):
    def get_action(policy, obs):
        logits = policy(obs)
        return Categorical(logits=logits).sample().item()

    obs, _ = env.reset()       # first obs comes from starting distribution
    done = False               # signal from environment that episode is over

    cum_ret = 0
    obs_traj = []
    while not done:

        if rendering:
            env.render()

        act = get_action(policy, torch.as_tensor(obs, dtype=torch.float32))
        obs, rew, terminated, truncated, _ = env.step(act)
        done = terminated or truncated
        cum_ret += rew
        obs_traj.append(obs)

    return obs_traj, cum_ret


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', '--env', type=str, default='CartPole-v0')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--checkpoint', type=str, default='', help="pretrained policy weights")
    parser.add_argument('--num_rollouts', type=int, default=1)

    args = parser.parse_args()
    checkpoint = args.checkpoint

    env = gym.make(args.env_name)
    obs_dim = env.observation_space.shape[0]
    n_acts = env.action_space.n
    hidden_sizes = [32]
    policy = mlp(sizes=[obs_dim] + hidden_sizes + [n_acts])
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if args.checkpoint == '':
        avg_returns = []
        checkpoints = np.arange(50)
        for i in checkpoints:
            ckpt_path = f"rlhf/policy_checkpoint{i}.params"
            policy.load_state_dict(torch.load(ckpt_path, map_location=device))

            returns = 0
            for _ in range(args.num_rollouts):
                _, cum_ret = generate_rollout(policy, env, rendering=False)
                returns += cum_ret

            avg_ret = returns / args.num_rollouts
            avg_returns.append(avg_ret)
            print(f"checkpoint {i}: average return {avg_ret:.2f}")

        plt.plot(checkpoints, avg_returns)
        plt.xlabel("Checkpoint")
        plt.ylabel("Average Return")
        plt.title("Average Return vs Checkpoint")
        plt.grid(True)
        plt.savefig("average_return_vs_checkpoint.png")
    else:
        policy.load_state_dict(torch.load(checkpoint, map_location=device))

        returns = 0
        for _ in range(args.num_rollouts):
            _, cum_ret = generate_rollout(policy, env, rendering=args.render)
            print("cumulative return", cum_ret)
            returns += cum_ret

        print("average return", returns / args.num_rollouts)