import gymnasium as gym
import torch
import torch.nn as nn
import numpy as np
from rollout_policy import generate_rollout
from utils import mlp, RewardNetwork


def generate_reference_rollouts(env):
    checkpoints = [f"./synthetic/policy_checkpoint{i}.params" for i in range(10)]

    obs_dim = env.observation_space.shape[0]
    n_acts = env.action_space.n
    hidden_sizes = [32]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    demonstrations = []
    demo_returns = []

    for checkpoint in checkpoints:
        policy = mlp(sizes=[obs_dim] + hidden_sizes + [n_acts])
        policy.load_state_dict(torch.load(checkpoint, map_location=device))
        traj, ret = generate_rollout(policy, env)
        print("traj ground-truth return", ret)
        demonstrations.append(traj)
        demo_returns.append(ret)

    return demonstrations, demo_returns


def create_training_data(trajectories, cum_returns, num_pairs):
    training_pairs = []
    training_labels = []
    num_trajs = len(trajectories)

    for _ in range(num_pairs):
        ti = 0
        tj = 0
        while ti == tj:
            ti = np.random.randint(num_trajs)
            tj = np.random.randint(num_trajs)
        traj_i = trajectories[ti]
        traj_j = trajectories[tj]

        # Label based on cumulative returns
        if cum_returns[ti] > cum_returns[tj]:
            label = 0
        else:
            label = 1
        training_pairs.append((traj_i, traj_j))
        training_labels.append(label)

    return training_pairs, training_labels


def predict_traj_return(net, traj):
    traj = np.asarray(traj, dtype=np.float32)
    traj = torch.from_numpy(traj).float().to(next(net.parameters()).device)
    return net.predict_return(traj).item()


def learn_reward(reward_network, optimizer, training_inputs, training_outputs, num_iter, checkpoint_dir):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    loss_criterion = nn.CrossEntropyLoss()
    reward_network.train()
    for itr in range(num_iter):
        indices = np.random.permutation(len(training_inputs))
        total_loss = 0
        for i in indices:
            optimizer.zero_grad()
            traj_i, traj_j = training_inputs[i]
            label = training_outputs[i]
            traj_i = torch.tensor(np.array(traj_i), dtype=torch.float32, device=device)
            traj_j = torch.tensor(np.array(traj_j), dtype=torch.float32, device=device)
            ri = reward_network.predict_return(traj_i)
            rj = reward_network.predict_return(traj_j)
            logits = torch.stack([ri, rj], dim=0).unsqueeze(0)
            target = torch.tensor([label], dtype=torch.long, device=device)
            loss = loss_criterion(logits, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {itr+1}, Loss: {total_loss/len(indices)}")

    print("checkpointing")
    torch.save(reward_network.state_dict(), checkpoint_dir)
    print("finished training")


if __name__ == "__main__":
    env = gym.make("CartPole-v0")

    num_pairs = 20
    trajectories, traj_returns = generate_reference_rollouts(env)

    traj_pairs, traj_labels = create_training_data(trajectories, traj_returns, num_pairs)

    num_iter = 100
    lr = 0.001
    checkpoint = "./reward.params"

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    reward_net = RewardNetwork()
    reward_net.to(device)

    import torch.optim as optim
    optimizer = optim.Adam(reward_net.parameters(), lr=lr)
    learn_reward(reward_net, optimizer, traj_pairs, traj_labels, num_iter, checkpoint)

    print("performance on training data")
    for i, pair in enumerate(traj_pairs):
        trajA, trajB = pair
        print("predicted return trajA", predict_traj_return(reward_net, trajA))
        print("predicted return trajB", predict_traj_return(reward_net, trajB))
        if traj_labels[i] == 0:
            print("A should be better\n")
        else:
            print("B should be better\n")

