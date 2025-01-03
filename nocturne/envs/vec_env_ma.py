"""Vectorized environment wrapper for multi-agent environments."""
import logging
import time
from copy import deepcopy
from typing import Any, Dict, List

import gymnasium as gym
import numpy as np
from stable_baselines3.common.vec_env.base_vec_env import (
    VecEnv,
    VecEnvObs,
    VecEnvStepReturn,
)

from nocturne.envs.base_env import BaseEnv
from utils.config import load_config

logging.basicConfig(level=logging.INFO)


class MultiAgentAsVecEnv(VecEnv):
    """A wrapper that casts multi-agent environments as vectorized environments.

    Args:
    -----
        VecEnv (SB3 VecEnv): SB3 VecEnv base class.
    """

    def __init__(self, config, num_envs, psr=False):
        # Create Nocturne env
        self.env = BaseEnv(config)

        # Make action and observation spaces compatible with SB3 (requires gymnasium)
        self.action_space = gym.spaces.Discrete(self.env.action_space.n)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, self.env.observation_space.shape, np.float32)
        self.num_envs = num_envs  # The maximum number of agents allowed in the environmen
        self.psr = psr  # Whether to use PSR or not

        self.psr_dict = (
            self.init_scene_dict() if psr else None
        )  # Initialize dict to keep track of the average reward obtained in each scene
        self.n_episodes = 0
        self.episode_lengths = []
        self.rewards = []  # Log reward per step
        self.dead_agent_ids = []  # Log dead agents per step
        self.num_agents_collided = 0  # Keep track of how many agents collided
        self.num_agents_off_road = 0  # Keep track of how many agents went off road
        self.total_agents_in_rollout = 0  # Log total number of agents in rollout
        self.num_agents_goal_achieved = 0  # Keep track of how many agents reached their goal
        self.agents_in_scene = []
        self.filename = None  # If provided, always use the same file

    def _reset_seeds(self) -> None:
        """Reset all environments' seeds."""
        self._seeds = None

    def reset(self, seed=None):
        """Reset environment and return initial observations."""
        # Reset Nocturne env
        obs_dict = self.env.reset(self.filename, self.psr_dict)

        # Reset storage
        self.agent_ids = []
        self.rewards = []
        self.dead_agent_ids = []
        self.ep_veh_collisions = 0
        self.ep_off_road = 0    
        self.ep_goal_achieved = 0

        obs_all = np.full(fill_value=np.nan, shape=(self.num_envs, self.env.observation_space.shape[0]))
        for idx, agent_id in enumerate(obs_dict.keys()):
            self.agent_ids.append(agent_id)
            obs_all[idx, :] = obs_dict[agent_id]

        # Save obs in buffer
        self._save_obs(obs_all)

        logging.debug(f"RESET - agent ids: {self.agent_ids}")

        # Make dict for storing the last info set for each agent
        self.last_info_dicts = {agent_id: {} for agent_id in self.agent_ids}

        return self._obs_from_buf()

    def step(self, actions) -> VecEnvStepReturn:
        """Convert action vector to dict and call env.step()."""

        agent_actions = {
            agent_id: actions[idx] for idx, agent_id in enumerate(self.agent_ids) if agent_id not in self.dead_agent_ids
        }

        # Take a step to obtain dicts
        next_obses_dict, rew_dict, done_dict, info_dict = self.env.step(agent_actions)

        # Update dead agents based on most recent done_dict
        for agent_id, is_done in done_dict.items():
            if is_done and agent_id not in self.dead_agent_ids:
                self.dead_agent_ids.append(agent_id)
                # Store agents' last info dict
                self.last_info_dicts[agent_id] = info_dict[agent_id].copy()

        # Storage
        obs = np.full(fill_value=np.nan, shape=(self.num_envs, self.observation_space.shape[0]))
        self.buf_dones = np.full(fill_value=np.nan, shape=(self.num_envs,))
        self.buf_rews = np.full_like(self.buf_dones, fill_value=np.nan)
        self.buf_infos = [{} for _ in range(self.num_envs)]

        # Override NaN placeholder for each agent that is alive
        for idx, key in enumerate(self.agent_ids):
            if key in next_obses_dict:
                self.buf_rews[idx] = rew_dict[key]
                self.buf_dones[idx] = done_dict[key] * 1
                self.buf_infos[idx] = info_dict[key]
                obs[idx, :] = next_obses_dict[key]

        # Save step reward obtained across all agents
        self.rewards.append(sum(rew_dict.values()))
        self.agents_in_scene.append(len(self.agent_ids))

        # Store observation
        self._save_obs(obs)

        # Reset episode if ALL agents are done or we reach the max number of steps
        if done_dict["__all__"]:
            for agent_id in self.agent_ids:
                self.ep_veh_collisions += self.last_info_dicts[agent_id]["veh_veh_collision"] * 1
                self.ep_off_road += self.last_info_dicts[agent_id]["veh_edge_collision"] * 1
                self.ep_goal_achieved += self.last_info_dicts[agent_id]["goal_achieved"] * 1

            # Store the fraction of agents that collided in episode
            self.num_agents_collided += self.ep_veh_collisions
            self.num_agents_off_road += self.ep_off_road
            self.num_agents_goal_achieved += self.ep_goal_achieved
            self.total_agents_in_rollout += len(self.agent_ids)

            # Save final observation where user can get it, then reset
            for idx in range(len(self.agent_ids)):
                self.buf_infos[idx]["terminal_observation"] = obs[idx]

            # Log episode stats
            ep_len = self.step_num
            self.n_episodes += 1
            self.episode_lengths.append(ep_len)

            # Store reward at scene level
            if self.psr:
                self.psr_dict[self.env.file]["count"] += 1
                self.psr_dict[self.env.file]["reward"] += (sum(rew_dict.values())) / len(self.agent_ids)
                self.psr_dict[self.env.file]["goal_rate"] += self.ep_goal_achieved / len(self.agent_ids)

            # Reset
            obs = self.reset()

        return (
            self._obs_from_buf(),
            np.copy(self.buf_rews),
            np.copy(self.buf_dones),
            deepcopy(self.buf_infos),
        )

    def close(self) -> None:
        """Close the environment."""
        self.env.close()

    def init_scene_dict(self):
        """Create a dictionary of scenes and the average normalized reward obtained in each scene."""
        psr_dict = {}
        for scene_name in self.env.files:
            psr_dict[scene_name] = {"count": 0, "prob": 1 / len(self.env.files), "reward": 0, "goal_rate": 0}
        return psr_dict

    def reset_scene_dict(self):
        for scene_name in self.psr_dict.keys():
            self.psr_dict[scene_name]["count"] = 0
            self.psr_dict[scene_name]["reward"] = 0
            self.psr_dict[scene_name]["goal_rate"] = 0

    @property
    def step_num(self) -> List[int]:
        """The episodic timestep."""
        return self.env.step_num

    @property
    def render(self) -> List[int]:
        """The episodic timestep."""
        img = self.env.render()
        return img

    @property
    def ego_state_feat(self) -> int:
        """The dimension of the ego state."""
        return self.env.ego_state_feat

    @property
    def road_obj_feat(self) -> int:
        """The dimension of the road objects."""
        return self.env.road_obj_feat

    @property
    def road_graph_feat(self) -> int:
        """The dimension of the road points."""
        return self.env.road_graph_feat

    @property
    def stop_sign_feat(self) -> int:
        """The dimension of the stop signs."""
        return self.env.stop_sign_feat

    @property
    def tl_feat(self) -> int:
        """The dimension of the traffic lights."""
        return self.env.tl_feat

    def seed(self, seed=None):
        """Set the random seeds for all environments."""
        if seed is None:
            # To ensure that subprocesses have different seeds,
            # we still populate the seed variable when no argument is passed
            seed = int(np.random.randint(0, np.iinfo(np.uint32).max, dtype=np.uint32))

        self._seeds = [seed + idx for idx in range(self.num_envs)]
        return self._seeds

    def _save_obs(self, obs: VecEnvObs) -> None:
        """Save observations into buffer."""
        self.buf_obs = obs

    def _obs_from_buf(self) -> VecEnvObs:
        """Get observation from buffer."""
        return np.copy(self.buf_obs)

    def get_attr(self, attr_name, indices=None):
        raise NotImplementedError()

    def set_attr(self, attr_name, value, indices=None) -> None:
        raise NotImplementedError()

    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        raise NotImplementedError()

    def env_is_wrapped(self, wrapper_class, indices=None):
        raise NotImplementedError()

    def step_async(self, actions: np.ndarray) -> None:
        raise NotImplementedError()

    def step_wait(self) -> VecEnvStepReturn:
        raise NotImplementedError()


if __name__ == "__main__":
    MAX_AGENTS = 2
    NUM_STEPS = 400

    # Load environment variables and config
    env_config = load_config("env_config")

    # Set the number of max vehicles
    env_config.max_num_vehicles = MAX_AGENTS

    # Make environment
    env = MultiAgentAsVecEnv(config=env_config, num_envs=MAX_AGENTS)

    obs = env.reset()
    for global_step in range(NUM_STEPS):
        # Take random action(s) -- you'd obtain this from a policy
        actions = np.array([env.env.action_space.sample() for _ in range(MAX_AGENTS)])

        # Step
        obs, rew, done, info = env.step(actions)

        # Log
        logging.info(f"step_num: {env.step_num} (global = {global_step}) | done: {done} | rew: {rew}")

        time.sleep(0.2)
