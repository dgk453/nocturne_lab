# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Default Nocturne env with minor adaptations."""
import json
import logging
import random
import glob
import os
from collections import defaultdict, deque
from enum import Enum
from itertools import islice, product
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, TypeVar, Union

import numpy as np
import torch
from box import Box as ConfigBox
from gym import Env
from gym.spaces import Box, Discrete

from nocturne import Action, Simulation, Vector2D, Vehicle
from utils.config import load_config
np.set_printoptions(suppress=True)

_MAX_NUM_TRIES_TO_FIND_VALID_VEHICLE = 100

ActType = TypeVar("ActType")  # pylint: disable=invalid-name
ObsType = TypeVar("ObsType")  # pylint: disable=invalid-name
RenderType = TypeVar("RenderType")  # pylint: disable=invalid-name

class CollisionType(Enum):
    """Enum for collision types."""

    NONE = 0
    VEHICLE_VEHICLE = 1
    VEHICLE_EDGE = 2


class BaseEnv(Env):  # pylint: disable=too-many-instance-attributes
    """Nocturne base Gym environment."""

    def __init__(  # pylint: disable=too-many-arguments
        self,
        config: Dict[str, Any],
        *,
        img_width=1200,
        img_height=1200,
        draw_target_positions=True,
        padding=10.0,
    ) -> None:
        """Initialize a Nocturne environment.

        Args:
        ----
            config (dict): configuration file for the environment.

        Optional Args
        -------------
            img_width (int): width of the image to render.
            img_height (int): height of the image to render.
            draw_target_positions (bool): whether to draw the target positions.
            padding (float): padding to add to the image.
        """
        super().__init__()
        self.config = ConfigBox(config)
        self.config.data_path = Path(self.config.data_path)
        self._render_settings = {
            "img_width": img_width,
            "img_height": img_height,
            "draw_target_positions": draw_target_positions,
            "padding": padding,
        }
        self.seed(self.config.seed)

        self.use_av_only = self.config.use_av_only

        self.count_invalid = 0
        self.count_total = 0

        # Load valid vehicles dict 
        with open(self.config.data_path / "valid_files.json", encoding="utf-8") as file:
            self.valid_veh_dict = json.load(file)
      
        # Load files
        file_paths = self.config.data_path.glob('*.json')

        if self.config.fix_file_order:
            files = sorted([os.path.basename(file) for file in file_paths])
        else:
            files = [os.path.basename(file) for file in file_paths]
            random.shuffle(files)

        # Select subset of files to sample from
        if self.config.num_files != -1:
            self.files = files[: self.config.num_files]
        if len(self.files) == 0:
            raise ValueError("Data path does not contain scenes.")

        # Set observation space
        obs_dim = self._get_obs_space_dim()
        self.observation_space = Box(
            low=-np.inf,
            high=np.inf,
            shape=obs_dim,
        )

        # Set action space
        if self.config.discretize_actions:
            self._set_discrete_action_space()
        else:
            self._set_continuous_action_space()

        # Count total and invalid samples
        self.invalid_samples = 0
        self.total_samples = 0

    def apply_actions(self, action_dict: Dict[int, ActType]) -> None:
        """Apply a dict of actions to the vehicle objects.

        Args:
        ----
            action_dict (Dict[int, ActType]): Dictionary of actions to apply to the vehicles.
        """
        for veh_obj in self.scenario.getObjectsThatMoved():
            action = action_dict.get(veh_obj.id, None)
            if action is None:
                continue
            _apply_action_to_vehicle(veh_obj, action, idx_to_actions=self.idx_to_actions)

    def step(  # pylint: disable=arguments-renamed,too-many-locals,too-many-branches,too-many-statements
        self, action_dict: Dict[int, ActType]
    ) -> Tuple[Dict[int, ObsType], Dict[int, float], Dict[int, bool], Dict[int, Dict[str, Union[bool, str]]]]:
        """Run one timestep of the environment's dynamics.

        Args:
        ----
            action_dict (Dict[int, ActType]): Dictionary of actions to apply to the vehicles.

        Raises:
        ------
            ValueError: If the action is not of a supported type or if the vehicle collision type is unknown.


        Returns:
        -------
            Dict[int, ObsType]: Dictionary with observation for each vehicle.
            Dict[int, float]: Dictionary with reward for each vehicle.
            Dict[int, bool]: Dictionary with done flag for each vehicle.
            Dict[int, Dict[str, Union[bool, str]]]]: Dictionary with info for each vehicle.
        """
        obs_dict = {}
        rew_dict = {}
        done_dict = {}
        info_dict = defaultdict(dict)
        rew_cfg = self.config.rew_cfg
        
        # Apply actions for the controlled vehicles
        self.apply_actions(action_dict)
        
        # Set the vehicles we are not controlling in expert-control mode...
        # to make sure they step their fixed trajectories
        objects_to_teleport = [
            obj for obj in self.scenario.getVehicles()
            if obj in self.scenario.getObjectsThatMoved()
            and obj not in self.controlled_vehicles
        ]
        for obj in objects_to_teleport:
            obj.expert_control = True
            
        # Step the simulator
        self.simulation.step(self.config.dt)
        self.t += self.config.dt
        self.step_num += 1

        logging.debug(f'stepping {[obj.id for obj in objects_to_teleport]} ({len(objects_to_teleport)} / {len(objects_to_teleport)+len(self.controlled_vehicles)}) vehs in expert-control mode.\n')
        logging.debug(f'controlling vehicle(s): {[veh.id for veh in self.controlled_vehicles]}, is_av? {[veh.is_av for veh in self.controlled_vehicles]}, expert-control = {[veh.expert_control for veh in self.controlled_vehicles]}.')
        
        # Take actions for the controlled vehicles
        for veh_obj in self.controlled_vehicles:
            veh_id = veh_obj.getID()
            if veh_id in self.done_ids:
                continue
      
            if veh_obj.position.x == self.config.scenario.invalid_position:
                logging.debug(f"(IN STEP) at t = {self.step_num} in {self.file}, vehicle {veh_obj.id} is invalid (pos = {veh_obj.position.x}). Removing it.")
                self.invalid_samples += 1

            # Get vehicle observation
            self.context_dict[veh_id].append(self.get_observation(veh_obj))
            if self.config.subscriber.n_frames_stacked > 1:
                veh_deque = self.context_dict[veh_id]
                context_list = list(
                    islice(
                        veh_deque,
                        len(veh_deque) - self.config.subscriber.n_frames_stacked,
                        len(veh_deque),
                    )
                )
                obs_dict[veh_id] = np.concatenate(context_list)
            else:
                obs_dict[veh_id] = self.context_dict[veh_id][-1]
            rew_dict[veh_id] = 0
            done_dict[veh_id] = False
            info_dict[veh_id]["goal_achieved"] = False
            info_dict[veh_id]["collided"] = False
            info_dict[veh_id]["veh_veh_collision"] = False
            info_dict[veh_id]["veh_edge_collision"] = False
       
            obj_pos = veh_obj.position
            goal_pos = veh_obj.target_position

            ############################################
            #   Compute rewards
            ############################################
            position_target_achieved = True
            speed_target_achieved = True
            heading_target_achieved = True
            if rew_cfg.position_target:
                position_target_achieved = (goal_pos - obj_pos).norm() < rew_cfg.position_target_tolerance
            if rew_cfg.speed_target:
                speed_target_achieved = np.abs(veh_obj.speed - veh_obj.target_speed) < rew_cfg.speed_target_tolerance
            if rew_cfg.heading_target:
                heading_target_achieved = (
                    np.abs(_angle_sub(veh_obj.heading, veh_obj.target_heading)) < rew_cfg.heading_target_tolerance
                )
            if position_target_achieved and speed_target_achieved and heading_target_achieved:
                info_dict[veh_id]["goal_achieved"] = True
                rew_dict[veh_id] += rew_cfg.goal_achieved_bonus / rew_cfg.reward_scaling
        
            if rew_cfg.shaped_goal_distance and rew_cfg.position_target:
                # penalize the agent for its distance from goal
                # we scale by goal_dist_normalizers to ensure that this value is always
                # less than the penalty for collision
                if rew_cfg.goal_distance_penalty:
                    rew_dict[veh_id] -= (
                        rew_cfg.shaped_goal_distance_scaling
                        * ((goal_pos - obj_pos).norm() / self.goal_dist_normalizers[veh_id] + 1e4)
                        / rew_cfg.reward_scaling
                    )
                else:
                    # the minus one is to ensure that it's not beneficial to collide
                    # we divide by goal_achieved_bonus / episode_length to ensure that
                    # acquiring the maximum "get-close-to-goal" reward at every
                    # time-step is always less than just acquiring the goal reward once
                    rew_dict[veh_id] += (
                        rew_cfg.shaped_goal_distance_scaling
                        * (1 - (goal_pos - obj_pos).norm() / (self.goal_dist_normalizers[veh_id] + 1e4))
                        / rew_cfg.reward_scaling
                    )
                # repeat the same thing for speed and heading
                if rew_cfg.shaped_goal_distance and rew_cfg.speed_target:
                    if rew_cfg.goal_distance_penalty:
                        rew_dict[veh_id] -= (
                            rew_cfg.shaped_goal_distance_scaling
                            * (np.abs(veh_obj.speed - veh_obj.target_speed) / rew_cfg.goal_speed_scaling)
                            / rew_cfg.reward_scaling
                        )
                    else:
                        rew_dict[veh_id] += (
                            rew_cfg.shaped_goal_distance_scaling
                            * (1 - np.abs(veh_obj.speed - veh_obj.target_speed) / rew_cfg.goal_speed_scaling)
                            / rew_cfg.reward_scaling
                        )
                if rew_cfg.shaped_goal_distance and rew_cfg.heading_target:
                    if rew_cfg.goal_distance_penalty:
                        rew_dict[veh_id] -= (
                            rew_cfg.shaped_goal_distance_scaling
                            * (np.abs(_angle_sub(veh_obj.heading, veh_obj.target_heading)) / (2 * np.pi))
                            / rew_cfg.reward_scaling
                        )
                    else:
                        rew_dict[veh_id] += (
                            rew_cfg.shaped_goal_distance_scaling
                            * (1 - np.abs(_angle_sub(veh_obj.heading, veh_obj.target_heading)) / (2 * np.pi))
                            / rew_cfg.reward_scaling
                        )
            ############################################
            #   Handle potential done conditions
            ############################################
            # achieved our goal
            if info_dict[veh_id]["goal_achieved"] and self.config.get("remove_at_goal", True):
                done_dict[veh_id] = True
            if veh_obj.getCollided():
                info_dict[veh_id]["collided"] = True
                if int(veh_obj.collision_type) == CollisionType.VEHICLE_VEHICLE.value:
                    info_dict[veh_id]["veh_veh_collision"] = True
                elif int(veh_obj.collision_type) == CollisionType.VEHICLE_EDGE.value:
                    info_dict[veh_id]["veh_edge_collision"] = True
                elif int(veh_obj.collision_type) != CollisionType.NONE.value:
                    raise ValueError(f"Unknown collision type: {veh_obj.collision_type}.")
                rew_dict[veh_id] -= np.abs(rew_cfg.collision_penalty) / rew_cfg.reward_scaling
                if self.config.get("remove_at_collide", True):
                    done_dict[veh_id] = True
            # remove the vehicle so that its trajectory doesn't continue. This is
            # important in the multi-agent setting.
            if done_dict[veh_id]:
                self.done_ids.append(veh_id)
                if (info_dict[veh_id]["goal_achieved"] and self.config.get("remove_at_goal", True)) or (
                    info_dict[veh_id]["collided"] and self.config.get("remove_at_collide", True)
                ):
                    self.scenario.removeVehicle(veh_obj)

        if self.config.rew_cfg.shared_reward:
            total_reward = np.sum(rew_dict.values())
            rew_dict = {key: total_reward for key in rew_dict}

        if self.step_num >= self.config.episode_length:
            done_dict = {key: True for key in done_dict}

        done_dict["__all__"] = all(done_dict.values())

        self.total_samples += len(obs_dict.keys())

        return obs_dict, rew_dict, done_dict, info_dict

    def reset(  # pylint: disable=arguments-differ,too-many-locals,too-many-branches,too-many-statements
        self,
        filename=None,
        psr_dict=None,
    ) -> Dict[int, ObsType]:
        """Reset the environment.

        Args:
        ----
        filename: If provided, reset env to this traffic scene.
        psr_dict: If provided, reset env to a scene sampled with given probabilities.

        Returns:
        -------
            Dict[int, ObsType]: Dictionary of observations for each vehicle.
        """
        self.t = 0
        self.step_num = 0

        # we don't want to initialize scenes with 0 actors after satisfying
        # all the conditions on a scene that we have

        for _ in range(_MAX_NUM_TRIES_TO_FIND_VALID_VEHICLE):
            # RESET TO NEW TRAFFIC SCENE
            if filename is not None:
                # Reset to a specific scene name
                self.file = filename
            elif self.config.sample_file_method == "no_replacement":
                # Random uniformly without replacement
                self.file = self.files.pop()
            elif psr_dict is not None:
                # Prioritized scene replay: sample according to probabilities
                probs = [item["prob"] for item in psr_dict.values()]
                self.file = np.random.choice(self.files, p=probs)
            else:  # Random uniformly with replacement (default)
                self.file = np.random.choice(self.files)

            self.simulation = Simulation(str(self.config.data_path / self.file), config=self.config.scenario)
            
            self.scenario = self.simulation.getScenario()

            # Get controlled vehicles
            if self.use_av_only:  # Control only the AVs
                self.controlled_vehicles = []
                all_vehs = self.scenario.getObjectsThatMoved()
                all_vehs.extend(self.scenario.getVehicles())
                all_vehs = list(set(all_vehs))
                for vehicle in all_vehs:
                    if vehicle.is_av:
                        self.controlled_vehicles.append(vehicle)
                    else:  # Put in expert control mode
                        vehicle.expert_control = True
                
                if len(self.controlled_vehicles) == 0:
                    raise ValueError(f"Scene {self.file!s} has no AV vehicles in. Skip")

            #####################################################################
            #   Construct context dictionary of observations that can be used to
            #   warm up policies by stepping all vehicles as experts.
            #####################################################################
            dead_feat = -np.ones(
                self.get_observation(self.scenario.getVehicles()[0]).shape[0] * self.config.subscriber.n_frames_stacked
            )
            # Step all the vehicles forward by one second and record their observations
            # as context
            self.config.scenario.context_length = max(
                self.config.scenario.context_length, self.config.subscriber.n_frames_stacked
            )  # Note: Consider raising an error if context_length < n_frames_stacked.
            self.context_dict = {
                veh.getID(): deque(
                    [dead_feat for _ in range(self.config.scenario.context_length)],
                    maxlen=self.config.scenario.context_length,
                )
                for veh in self.scenario.getVehicles()
            }
            for veh in self.scenario.getObjectsThatMoved():
                veh.expert_control = True
            for _ in range(self.config.scenario.context_length):
                for veh in self.scenario.getObjectsThatMoved():
                    obs = self.get_observation(veh)
                    self.context_dict[veh.getID()].append(obs)
                # Step simulator
                self.simulation.step(self.config.dt)
                # Make sure to increment counter
                self.step_num += 1
            # now hand back control to our actual controllers
            for veh in self.scenario.getObjectsThatMoved():
                veh.expert_control = False

            # remove all the objects that are in collision or are already in goal dist
            # additionally set the objects that have infeasible goals to be experts
            for veh_obj in self.simulation.getScenario().getObjectsThatMoved():
                obj_pos = _position_as_array(veh_obj.getPosition())
                goal_pos = _position_as_array(veh_obj.getGoalPosition())

                ############################################
                #    Remove vehicles at goal
                ############################################
                norm = np.linalg.norm(goal_pos - obj_pos)
                if norm < self.config.rew_cfg.goal_tolerance or veh_obj.getCollided():
                    self.scenario.removeVehicle(veh_obj)
                ############################################
                #    Set all vehicles with unachievable goals to be experts
                ############################################
                if self.file in self.valid_veh_dict and veh_obj.getID() in self.valid_veh_dict[self.file]:
                    veh_obj.expert_control = True
            ############################################
            #    Pick out the vehicles that we are controlling
            ############################################
            if not self.use_av_only:  # No restrictions on which vechicles can be controlled
                # Ensure that no more than max_num_vehicles are controlled
                temp_vehicles = np.random.permutation(self.scenario.getObjectsThatMoved())
                curr_index = 0
                self.controlled_vehicles = []

                for vehicle in temp_vehicles:
                    # Remove vehicles that have invalid positions
                    veh_at_invalid_pos = np.isclose(
                        vehicle.position.x,
                        self.config.scenario.invalid_position,
                    )

                    # Exclude vehicles with invalid goal positions
                    veh_has_invalid_goal_pos = np.isclose(
                        vehicle.getGoalPosition().x, self.config.scenario.invalid_position
                    ) or np.isclose(vehicle.getGoalPosition().y, self.config.scenario.invalid_position)

                    if veh_at_invalid_pos or veh_has_invalid_goal_pos:
                        self.scenario.removeVehicle(vehicle)

                    # Otherwise the vehicle is valid and we add it to the list of controlled vehicles
                    if (
                        not vehicle.expert_control
                        and not veh_at_invalid_pos
                        and not veh_has_invalid_goal_pos
                        and curr_index < self.config.max_num_vehicles
                    ):
                        self.controlled_vehicles.append(vehicle)
                        curr_index += 1
                    else:
                        vehicle.expert_control = True

            self.all_vehicle_ids = {veh.getID(): veh for veh in self.controlled_vehicles}

            # check that we have at least one vehicle or if we have just one file, exit anyways
            # or else we might be stuck in an infinite loop
            if len(self.all_vehicle_ids) > 0:
                break
        else:  # No break in for-loop, i.e., no valid vehicle found in any of the files.
            raise ValueError(f"No controllable vehicles in any of the {len(self.files)} scenes.")

        # Set goal positions for controlled vehicles
        self._set_goal_positions()

        # Construct the observations and goal normalizers
        obs_dict = {}
        self.goal_dist_normalizers = {}
        max_goal_dist = -np.inf
        for veh_obj in self.controlled_vehicles:
            veh_id = veh_obj.getID()
            # store normalizers for each vehicle
            obj_pos = _position_as_array(veh_obj.getPosition())
            goal_pos = _position_as_array(veh_obj.getGoalPosition())
            dist = np.linalg.norm(obj_pos - goal_pos)
            self.goal_dist_normalizers[veh_id] = dist
            # compute the obs
            obs = self.get_observation(veh_obj)
            self.context_dict[veh_id].append(obs)
            if self.config.subscriber.n_frames_stacked > 1:
                veh_deque = self.context_dict[veh_id]
                context_list = list(
                    islice(
                        veh_deque,
                        len(veh_deque) - self.config.subscriber.n_frames_stacked,
                        len(veh_deque),
                    )
                )
                obs_dict[veh_id] = np.concatenate(context_list)
            else:
                obs_dict[veh_id] = self.context_dict[veh_id][-1]
            # pick the vehicle that has to travel the furthest distance and use it for
            # rendering
            if dist > max_goal_dist:
                # this attribute is just used for rendering of the view
                # from the ego frame
                self.render_vehicle = veh_obj
                max_goal_dist = dist

        self.done_ids = []
        
        # Sanity check: Check if any vehicle is at an invalid position
        for veh_id in obs_dict.keys():
            veh_obj = self.all_vehicle_ids[veh_id]
            if np.isclose(veh_obj.position.x, self.config.scenario.invalid_position):
                logging.debug(f"obs_dict contains invalid vehicle! veh_id: {veh_id} at t = {self.step_num}")
                logging.debug(f"obs_max: {obs_dict[veh_id].max()}")
                self.invalid_samples += 1

        self.total_samples += len(obs_dict.keys())

        return obs_dict

    def get_observation(self, veh_obj: Vehicle) -> np.ndarray:
        """Return the observation for a particular vehicle.

        Args:
        ----
            veh_obj (Vehicle): Vehicle object to get the observation for.

        Returns:
        -------
            np.ndarray: Observation for the vehicle.
        """
        cur_position = []
        if self.config.subscriber.use_current_position:
            cur_position = _position_as_array(veh_obj.getPosition())
            speed = np.array([veh_obj.getSpeed()])
            steer = np.array([veh_obj.steering])
            if self.config.normalize_state:
                cur_position = cur_position / np.linalg.norm(cur_position)

            cur_position = np.concatenate([cur_position, speed, steer])

        ego_state = []
        if self.config.subscriber.use_ego_state:
            ego_state = self.scenario.ego_state(veh_obj)

            if self.config.normalize_state:
                ego_state_norm = self.normalize_ego_state_by_cat(ego_state)
            
        visible_state = []
        if self.config.subscriber.use_observations:
            visible_state = self.scenario.flattened_visible_state(
                veh_obj, self.config.subscriber.view_dist, self.config.subscriber.view_angle
            )
            if self.config.normalize_state:
                visible_state = self.normalize_obs_by_cat(visible_state)

        # Concatenate
        obs = np.concatenate((ego_state_norm, visible_state, cur_position))
        
        return obs

    def _get_obs_space_dim(self, base=0):
        """Calculate observation dimension based on the configs."""
        # Set dimensions (fixed values)
        self.road_obj_feat = 13
        self.road_graph_feat = 13
        self.stop_sign_feat = 3
        self.tl_feat = 12
        self.ego_state_feat = 10

        # Compute observation dimension
        obs_space_dim = 0

        if self.config.subscriber.use_ego_state:
            obs_space_dim += self.ego_state_feat

        if self.config.subscriber.use_current_position:
            obs_space_dim += 2

        if self.config.subscriber.use_observations:
            self.ro_dim = self.road_obj_feat * self.config.scenario.max_visible_objects
            self.rg_dim = self.road_graph_feat * self.config.scenario.max_visible_road_points
            self.tl_dim = self.tl_feat * self.config.scenario.max_visible_traffic_lights
            self.ss_dim = self.stop_sign_feat * self.config.scenario.max_visible_stop_signs

            obs_space_dim += base + self.ro_dim + self.rg_dim + self.tl_dim + self.ss_dim

        # Multiply by memory to get the final dimension
        obs_space_dim = obs_space_dim * self.config.subscriber.n_frames_stacked

        return (obs_space_dim,)

    def _set_goal_positions(self) -> None:
        """Set the goal positions for the controlled vehicles."""
        for veh_obj in self.controlled_vehicles:
            # Set the goal position to a random position on the expert trajectory
            if self.config.target_positions.randomize_goals:
                # Create list with intermediate goal positions
                goal_positions = [
                    self.scenario.expert_position(veh_obj, goal_time_step)
                    for goal_time_step in self.config.target_positions.time_steps_list
                ]
                # Add original goal position
                goal_positions += [veh_obj.target_position]

                # Remove invalid goal positions (experts are done at different steps)
                goal_positions = [
                    goal_pos for goal_pos in goal_positions if goal_pos.x != self.config.scenario.invalid_position
                ]

                # Sample random position
                rand_goal_pos = random.choice(goal_positions)

                # Set vehicle goal position
                veh_obj.setGoalPosition(rand_goal_pos)

            else:  # Keep the standard goal positions at the end of the expert trajectory
                veh_obj.setGoalPosition(veh_obj.target_position)

    def normalize_ego_state_by_cat(self, state):
        """Divide every feature in the ego state by the maximum value of that feature."""
        return state / (np.array([float(val) for val in self.config.ego_state_feat_max.values()]))

    def normalize_obs_by_cat(self, state):
        """Divide all visible state elements by the maximum value across the visible state."""
        return state / self.config.vis_obs_max

    def render(self) -> Optional[RenderType]:  # pylint: disable=unused-argument
        """Render the environment.

        Args:
        ----
            mode (Optional[bool]): Render mode.

        Returns:
        -------
            Optional[RenderType]: Rendered image.
        """
        return self.scenario.getImage(**self._render_settings)

        env.scenario.getImage(
            **self._render_settings,
        )

    def render_ego(self) -> Optional[RenderType]:  # pylint: disable=unused-argument
        """Render the ego vehicles.

        Args:
        ----
            mode (Optional[bool]): Render mode.

        Returns:
        -------
            Optional[RenderType]: Rendered image.
        """
        if self.render_vehicle.getID() in self.done_ids:
            return None
        return self.scenario.getConeImage(
            source=self.render_vehicle,
            view_dist=self.config.subscriber.view_dist,
            view_angle=self.config.subscriber.view_angle,
            head_angle=self.render_vehicle.head_angle,
            **self._render_settings,
        )

    def render_features(self) -> Optional[RenderType]:  # pylint: disable=unused-argument
        """Render the features.

        Args:
        ----
            mode (Optional[bool]): Render mode.

        Returns:
        -------
            Optional[RenderType]: Rendered image.
        """
        if self.render_vehicle.getID() in self.done_ids:
            return None
        return self.scenario.getFeaturesImage(
            source=self.render_vehicle,
            view_dist=self.config.subscriber.view_dist,
            view_angle=self.config.subscriber.view_angle,
            head_angle=self.render_vehicle.head_angle,
            **self._render_settings,
        )

    def seed(self, seed: Optional[int] = None) -> None:
        """Seed the environment.

        Args:
        ----
            seed (Optional[int]): Seed to use.
        """
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

    def _set_discrete_action_space(self) -> None:
        """Set the discrete action space."""
        self.action_space = Discrete(self.config.accel_discretization * self.config.steering_discretization)
        self.accel_grid = np.linspace(
            -np.abs(self.config.accel_lower_bound),
            self.config.accel_upper_bound,
            self.config.accel_discretization,
        )
        self.steering_grid = np.linspace(
            -np.abs(self.config.steering_lower_bound),
            self.config.steering_upper_bound,
            self.config.steering_discretization,
        )
        self.idx_to_actions = {}
        self.actions_to_idx = {}
        for i, (accel, steer) in enumerate(product(self.accel_grid, self.steering_grid)):
            self.idx_to_actions[i] = [accel, steer]
            self.actions_to_idx[accel, steer] = [i]

    def _set_continuous_action_space(self) -> None:
        """Set the continuous action space."""
        self.action_space = Box(
            low=-np.array(
                [
                    np.abs(self.config.accel_lower_bound),
                    self.config.steering_lower_bound,
                ]
            ),
            high=np.array(
                [
                    np.abs(self.config.accel_upper_bound),
                    self.config.steering_upper_bound,
                ]
            ),
        )
        self.idx_to_actions = None

    def unflatten_obs(self, obs_flat):
        "Unsqueeeze the flattened object." ""

        # OBS FLAT ORDER: road_objects, road_points, traffic_lights, stop_signs
        # Find the ends of each section
        road_objects_end = 13 * self.config.scenario.max_visible_objects
        road_points_end = road_objects_end + (13 * self.config.scenario.max_visible_road_points)
        tl_end = road_points_end + (12 * self.config.scenario.max_visible_traffic_lights)
        stop_sign_end = tl_end + (3 * self.config.scenario.max_visible_stop_signs)

        # Unflatten
        road_objects = obs_flat[:road_objects_end]
        road_points = obs_flat[road_objects_end:road_points_end]
        traffic_lights = obs_flat[road_points_end:tl_end]
        stop_signs = obs_flat[tl_end:stop_sign_end]

        return road_objects, road_points, traffic_lights, stop_signs


def _angle_sub(current_angle: float, target_angle: float) -> float:
    """Subtract two angles to find the minimum angle between them.

    Args:
    ----
        current_angle (float): Current angle.
        target_angle (float): Target angle.

    Returns:
    -------
        float: Minimum angle between the two angles.
    """
    # Subtract the angles, constraining the value to [0, 2 * np.pi)
    diff = (target_angle - current_angle) % (2 * np.pi)

    # If we are more than np.pi we're taking the long way around.
    # Let's instead go in the shorter, negative direction
    if diff > np.pi:
        diff = -(2 * np.pi - diff)
    return diff


def _apply_action_to_vehicle(
    veh_obj: Vehicle, action: ActType, *, idx_to_actions: Optional[Dict[int, Tuple[float, float]]] = None
) -> None:
    """Apply an action to a vehicle.

    Args:
    ----
        veh_obj (Vehicle): Vehicle object to apply the action to.
        action (ActType): Action to apply to the vehicle.

    Optional Args
    -------------
        idx_to_actions (Optional[Dict[int, Tuple[float, float]]]): Dictionary of actions to apply to the vehicle.

    Raises:
    ------
        NotImplementedError: If the action type is not supported.
    """
    if isinstance(action, Action):
        veh_obj.apply_action(action)
    elif isinstance(action, np.ndarray):
        veh_obj.apply_action(Action.from_numpy(action))
    elif isinstance(action, (tuple, list)):
        veh_obj.acceleration = action[0]
        veh_obj.steering = action[1]
    elif isinstance(action, int) and idx_to_actions is not None:
        accel, steer = idx_to_actions[action]
        veh_obj.acceleration = accel
        veh_obj.steering = steer
    else:
        accel, steer = idx_to_actions[action]
        veh_obj.acceleration = accel
        veh_obj.steering = steer


def _position_as_array(position: Vector2D) -> np.ndarray:
    """Convert a position to an array.

    Args:
    ----
        position (Vector2D): Position to convert.

    Returns:
    -------
        np.ndarray: Position as an array.
    """
    return np.array([position.x, position.y])
