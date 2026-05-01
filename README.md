# Preference Reward Learning for CartPole

This project is a compact example of preference-based reward learning on the CartPole environment. It trains a set of baseline policies, converts their rollouts into pairwise preferences, learns a neural reward model from those preferences, and then trains a new policy using the learned reward.

The code is intentionally small and script-oriented so the full workflow is easy to inspect.

## Features

- Vanilla policy gradient training for CartPole.
- Pairwise preference generation from rollout returns.
- Neural reward model trained with cross-entropy over trajectory preferences.
- Policy evaluation and checkpoint plotting utilities.

## Setup

Create and activate a Python environment, then install the dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows, activate the environment with:

```bash
.venv\Scripts\activate
```

## Usage

First, train a small collection of baseline policies. These checkpoints are used to generate reference rollouts:

```bash
python vpg.py --epochs 10 --checkpoint --checkpoint_dir ./synthetic
```

Next, train the reward model from pairwise preferences:

```bash
python offline_reward_learning.py
```

This writes the learned reward weights to `reward.params`.

Finally, train a policy against the learned reward model:

```bash
python vpg.py --epochs 50 --checkpoint --reward_params reward.params --checkpoint_dir ./rlhf
```

Evaluate a saved policy checkpoint with:

```bash
python rollout_policy.py --checkpoint ./rlhf/policy_checkpoint49.params --num_rollouts 5
```

To render the environment while evaluating, add `--render`.

## Project Structure

- `vpg.py`: vanilla policy gradient training loop.
- `offline_reward_learning.py`: preference-pair generation and reward model training.
- `rollout_policy.py`: checkpoint evaluation and rollout plotting.
- `utils.py`: shared neural-network helpers and the `RewardNetwork` model.
- `requirements.txt`: Python dependencies.

## Notes

The default environment is `CartPole-v0`. Generated files such as checkpoints, learned parameters, and plots are ignored by git.

## License

MIT
