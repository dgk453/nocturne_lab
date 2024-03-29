# `nocturne_lab`: A lightweight, multi-agent driving simulator 🧪 + 🚗

`nocturne_lab` is a maintained fork of [Nocturne](https://github.com/facebookresearch/nocturne); a 2D, partially observed, driving simulator built in C++. You can get started with the intro examples 🏎️💨 [here](https://github.com/Emerge-Lab/nocturne_lab/tree/feature/nocturne_fork_cleanup/examples).

---

> ### See our [project page](https://sites.google.com/view/driving-partners)

---

## Dataset 

You can download a part of the dataset (~2000 scenes) [here](https://www.dropbox.com/scl/fi/e5kjf7w8kxrop8ume7u2f/data.zip?rlkey=mix6dnexzdz48330p0m8s0r9s&dl=0). Once downloaded, add the data to the `./data` folder and make sure the `data_path` in `env_config` is set correctly.

## Algorithms

| Algorithm                  | Reference                                                     | Implementation                                                                                   | How to run                                                     |
| -------------------------- | ------------------------------------------------------------- | ------------------------------------------------------------------------------------------------ | -------------------------------------------------------------- |
| MAPPO                      | [(Vinitsky et al., 2021)](https://arxiv.org/abs/2103.01955)   | [ma_ppo.py](https://github.com/Emerge-Lab/nocturne_lab/blob/hr_rl/algorithms/ppo/sb3/ma_ppo.py)  | `python experiments/hr_rl/run_hr_ppo_cli.py --reg-weight 0.0`  |
| Human-Regularized (MA) PPO | [(Cornelisse et al., 2024)](https://arxiv.org/abs/2403.19648) | [reg_ppo.py](https://github.com/Emerge-Lab/nocturne_lab/blob/main/algorithms/ppo/sb3/reg_ppo.py) | `python experiments/hr_rl/run_hr_ppo_cli.py --reg-weight 0.06` |


## Trained policies 🏋️‍♂️

We release the best PPO-trained models with human regularization in [`models_trained/hr_rl`](https://github.com/Emerge-Lab/nocturne_lab/tree/hr_rl/models_trained/hr_rl). Additionally, we release the human reference policies, which can be found at [`models_trained/il`](https://github.com/Emerge-Lab/nocturne_lab/tree/hr_rl/models_trained/il). For the results presented in the paper, we used the IL policy trained on AVs (`human_policy_D651_S500_02_18_20_05_AV_ONLY.pt`).


## Run HR-PPO in 3 steps 🚀

After installing `nocturne_lab`, here is how you can run your own Human-Regularized PPO in 3 steps:

- **Step 1**: Make sure you installed the dataset and set the `data_path` in `configs/env_config.yaml` to your folder.
- **Step 2**: You have access to our trained imitation learning policy in [models_trained/il](https://github.com/Emerge-Lab/nocturne_lab/tree/hr_rl/models_trained/il). Make sure that the `human_policy_path` in the `configs/exp_config.yaml` file is set to the IL policy you want to use.
- **Step 3**: That's it! Now run:
```Python
python experiments/hr_rl/run_hr_ppo_cli.py --reg-weight <your-regularization-weight>
```
where setting `reg-weight 0.0` will just run standard MAPPO. We used a regularization weight between 0.02 - 0.08 for the paper.


## Basic RL interface

```python
from nocturne.envs.base_env import BaseEnv

# Initialize an environment
env = BaseEnv(config=env_config)

# Reset
obs_dict = env.reset()

# Get info
agent_ids = [agent_id for agent_id in obs_dict.keys()]
dead_agent_ids = []

for step in range(1000):

    # Sample actions
    action_dict = {
        agent_id: env.action_space.sample()
        for agent_id in agent_ids
        if agent_id not in dead_agent_ids
    }

    # Step in env
    obs_dict, rew_dict, done_dict, info_dict = env.step(action_dict)

    # Update dead agents
    for agent_id, is_done in done_dict.items():
        if is_done and agent_id not in dead_agent_ids:
            dead_agent_ids.append(agent_id)

    # Reset if all agents are done
    if done_dict["__all__"]:
        obs_dict = env.reset()
        dead_agent_ids = []

# Close environment
env.close()
```


## Installation
The instructions for installing Nocturne locally are provided below. To use the package on a HPC (e.g. HPC Greene), follow the instructions in [./hpc/hpc_setup.md](./hpc/hpc_setup.md).


### Requirements

* Python (>=3.10)

### Virtual environment
Below different options for setting up a virtual environment are described. Either option works although `pyenv` is recommended.

> _Note:_ The virtual environment needs to be **activated each time** before you start working.

#### Option 1: `pyenv`
Create a virtual environment by running:

```shell
pyenv virtualenv 3.10.12 nocturne_lab
```

The virtual environment should be activated every time you start a new shell session before running subsequent commands:

```shell
pyenv shell nocturne_lab
```

Fortunately, `pyenv` provides a way to assign a virtual environment to a directory. To set it for this project, run:
```shell
pyenv local nocturne_lab
```

#### Option 2: `conda`
Create a conda environment by running:

```shell
conda env create -f ./environment.yml
```

This creates a conda environment using Python 3.10 called `nocturne_lab`.

To activate the virtual environment, run:

```shell
conda activate nocturne_lab
```

#### Option 3: `venv`
Create a virtual environment by running:

```shell
python -m venv .venv
```

The virtual environment should be activated every time you start a new shell session before running the subsequent command:

```shell
source .venv/bin/activate
```

### Dependencies

`poetry` is used to manage the project and its dependencies. Start by installing `poetry` in your virtual environment:

```shell
pip install poetry
```

Before installing the package, you first need to synchronise and update the git submodules by running:

```shell
# Synchronise and update git submodules
git submodule sync
git submodule update --init --recursive
```

Now install the package by running:

```shell
poetry install
```

> _Note_: If it fails to build `nocturne`, try running `poetry build` to get a more descriptive error message. One reason it fails may be because you don't have SFML installed, which can be done by running `brew install sfml` on mac or `sudo apt-get install libsfml-dev` on Linux.

---
> Under the hood the `nocturne` package uses the `nocturne_cpp` Python package that wraps the Nocturne C++ code base and provides bindings for Python to interact with the C++ code using `pybind11`.
---

### Common errors

- `KeyringLocked Failed to unlock the collection!`. Solution: first run `export PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring` in your terminal, then rerun `poetry install` [stackOverflow with more info](https://stackoverflow.com/questions/74438817/poetry-failed-to-unlock-the-collection)

### Development setup
To configure the development setup, run:
```shell
# Install poetry dev dependencies
poetry install --only=dev

# Install pre-commit (for flake8, isort, black, etc.)
pre-commit install

# Optional: Install poetry docs dependencies
poetry install --only=docs
```
