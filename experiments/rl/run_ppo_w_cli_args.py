"""Cast a multi-agent env as vec env to use SB3's PPO."""
import logging
from datetime import datetime
from typing import List, Union
import typer
import torch
import wandb
import numpy as np

# Multi-agent as vectorized environment
from nocturne.envs.vec_env_ma import MultiAgentAsVecEnv
from utils.config import load_config
from utils.render import make_video

# Custom callback
from utils.sb3.callbacks import CustomMultiAgentCallback

# Custom PPO class that supports multi-agent control
from utils.sb3.ma_ppo import MultiAgentPPO
from utils.string_utils import datetime_to_str
from util.random_utils import init_seed

logging.basicConfig(level=logging.INFO)

# Default settings
env_config = load_config("env_config")
exp_config = load_config("exp_config")
video_config = load_config("video_config")

POLICY_SIZE_DICT = {
    "small": [128, 128], 
    "medium": [512, 256, 128], 
    "large": [1024, 512, 256, 128],
}
POLICY_TYPE_DICT = {
    "mlp": "MlpPolicy",
}

def run_ppo(
    sweep_name: str="ppo",
    steer_disc: int=5, 
    accel_disc: int=5, 
    ent_coef: float=0.,
    vf_coef: float=0.5,
    seed: int=42,
    policy_arch: str="mlp",
    policy_size: str="small",
    activation_fn: str="tanh",
    total_timesteps: int=100_000,
    num_files: int=10,
    single_scene: int=0,
) -> None:
    """Train RL agent using PPO with CLI arguments."""

    # ==== Update run params ==== #
    # Environment
    env_config.steer_disc = steer_disc
    env_config.accel_disc = accel_disc
    env_config.num_files = num_files
    # Experiment
    exp_config.ent_coef = ent_coef
    exp_config.vf_coef = vf_coef
    exp_config.learn.total_timesteps = total_timesteps
    exp_config.train_on_single_scene = single_scene
    exp_config.policy_arch = policy_arch
    exp_config.policy_size = policy_size
    exp_config.activation_func = activation_fn
    # Use custom policy 
    policy_type = POLICY_TYPE_DICT[policy_arch]
    policy_layers = POLICY_SIZE_DICT[policy_size]
    afunc = torch.nn.Tanh if activation_fn == "tanh" else torch.nn.ReLU
    policy_kwargs = dict( 
        activation_fn=afunc, 
        net_arch=policy_layers,
    )
    # ==== Update run params ==== #

    # Ensure reproducability
    init_seed(env_config, exp_config, seed)

    # Make environment
    env = MultiAgentAsVecEnv(
        config=env_config, 
        num_envs=env_config.max_num_vehicles,
        train_on_single_scene=exp_config.train_on_single_scene,
    )

    # Set up wandb
    RUN_ID = None
    if exp_config.track_wandb:
        
        # Set up run
        datetime_ = datetime_to_str(dt=datetime.now())
        RUN_ID = f"{exp_config.exp_name}_{datetime_}"
    
        # Add scene to config
        exp_config.scene = env.filename

        run = wandb.init(
            project=sweep_name,
            name=RUN_ID,
            config={**exp_config, **env_config},
            id=RUN_ID,
            **exp_config.wandb,
        )
    
    # Log basic exp info
    logging.info(f"Created env. Max # agents = {env_config.max_num_vehicles}.")
    logging.info(f"Learning in {env_config.num_files} scene(s): {env.env.files}")
    logging.info(f"Obs_space     : {env.observation_space.shape[0]} ---")
    logging.info(f"Action_space\n: {env.env.idx_to_actions}")

    # Set device
    exp_config.ppo.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize custom callback
    custom_callback = CustomMultiAgentCallback(
        env_config=env_config,
        exp_config=exp_config,
        video_config=video_config,
        wandb_run=run if RUN_ID is not None else None,
    )

    # Make scene init video to check expert actions
    if exp_config.track_wandb:
        for model in exp_config.wandb_init_videos:
            make_video(
                env_config=env_config,
                exp_config=exp_config,
                video_config=video_config,
                filenames=[env.filename],
                model=model,
                n_steps=None,
            )

    # Set up PPO model
    model = MultiAgentPPO(
        env=env,
        n_steps=exp_config.ppo.n_steps,
        policy=policy_type,
        policy_kwargs=policy_kwargs,
        ent_coef=exp_config.ppo.ent_coef,
        vf_coef=exp_config.ppo.vf_coef,
        seed=exp_config.seed,  # Seed for the pseudo random generators
        tensorboard_log=f"runs/{RUN_ID}" if RUN_ID is not None else None,
        verbose=exp_config.verbose,
        device=exp_config.ppo.device,
    )

    # Learn
    model.learn(
        **exp_config.learn,
        callback=custom_callback,
    )

    # Finish
    if exp_config.track_wandb:
        run.finish()


if __name__ == "__main__":

    # Run
    typer.run(run_ppo)