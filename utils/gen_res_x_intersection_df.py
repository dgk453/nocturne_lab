# Dependencies
import logging
import pandas as pd
import pickle
from utils.policies import load_policy
from evaluation.policy_evaluation import evaluate_policy
from utils.config import load_config
from utils.sb3.reg_ppo import RegularizedPPO
import pickle

# Load intersection info
with open('evaluation/scene_info/valid_all_01_23_11_18.pkl', 'rb') as handle:
    valid_scene_to_paths_dict = pickle.load(handle)

with open('evaluation/scene_info/train_1000_01_31_15_18.pkl', 'rb') as handle:
    train_scene_to_paths_dict = pickle.load(handle)
    
"""Evaluation settings."""
NUM_EVAL_EPISODES = 400 # Number of episodes to evaluate on
DETERMINISTIC = True 
SELECT_FROM = 50 # Number of scenes
MAX_CONTROLLED_AGENTS = 200 # All agents are controlled
METRICS = ['goal_rate', 'off_road', 'veh_veh_collision']

TRAIN_DATA_PATH = 'data_full/train'
#TEST_DATA_PATH = 'data_full/valid'

# Evaluation settings
EVAL_MODES = {'Self-play': 200, 'Log-replay': 1}
#DATASETS = {'Train': TRAIN_DATA_PATH, 'Test': TEST_DATA_PATH}
DATASETS = {'Train': TRAIN_DATA_PATH}

INTERSECTION_DICTS = {'Train': train_scene_to_paths_dict, 'Test': valid_scene_to_paths_dict}
H_COLUMNS = ['Agent', 'Train agent', 'Dataset', 'Eval mode']
METRICS = ['Off-Road Rate (\%)', 'Collision Rate (\%)', 'Goal Rate (\%)']

if __name__ == "__main__":
    
    # Load config files
    env_config = load_config("env_config")
    exp_config = load_config("exp_config")
    video_config = load_config("video_config")
    #models_config = load_config("model_config") # Trained models
    models_config = load_config("model_config_02") # Trained models
    
    # Evaluate HR-PPO policies
    df_ppo_all = pd.DataFrame()
    BASE_PATH = models_config.hr_ppo_models_dir_self_play
    for model_config in models_config.best_overall_models:
        for eval_mode in EVAL_MODES:
            for dataset in DATASETS:
                
                logging.info(f'Evaluate model {model_config.name} on {dataset} data using {eval_mode} mode\n')
                
                # Load policy
                policy = RegularizedPPO.load(
                    f'{BASE_PATH}/{model_config.name}'
                )
                
                # Set evaluation settings
                num_controlled_agents = EVAL_MODES[eval_mode]
                eval_dataset = DATASETS[dataset]
                scene_to_paths_dict = INTERSECTION_DICTS[dataset]
                
                if eval_mode == 'Self-play':
                    NUM_EVAL_EPISODES = 300
                elif eval_mode == 'Log-replay':
                    NUM_EVAL_EPISODES = 1500
                    
                # Evaluate policy
                df_res = evaluate_policy(
                    env_config=env_config,
                    scene_path_mapping=scene_to_paths_dict,
                    controlled_agents=num_controlled_agents,
                    data_path=eval_dataset,
                    mode='policy',
                    policy=policy,
                    select_from_k_scenes=SELECT_FROM,
                    num_episodes=NUM_EVAL_EPISODES,
                    deterministic=DETERMINISTIC,
                )
                
                # Add identifiers
                df_res['Agent'] = model_config.agent
                df_res['Train agent'] = model_config.train_agent
                df_res['Dataset'] = dataset
                df_res['Eval mode'] = eval_mode
                df_res['Reg. weight'] = model_config.reg_weight
                 
                # Store
                df_ppo_all = pd.concat([df_ppo_all, df_res], ignore_index=True)
                

    # Evaluate BC policy
    df_bc = pd.DataFrame()
    for bc_config in models_config.used_human_policy:
        for eval_mode in EVAL_MODES:
            for dataset in DATASETS:
                
                logging.info(f'Evaluate model {bc_config.name} on {dataset} data using {eval_mode} mode\n')
                
                # Load policy    
                human_policy = load_policy(
                    data_path=f"{models_config.bc_models_dir}",
                    file_name=bc_config.name, 
                )
                # Set evaluation settings
                num_controlled_agents = EVAL_MODES[eval_mode]
                eval_dataset = DATASETS[dataset]
                scene_to_paths_dict = INTERSECTION_DICTS[dataset]
                    
                # Evaluate policy
                df_res_bc = evaluate_policy(
                    env_config=env_config,
                    scene_path_mapping=scene_to_paths_dict,
                    controlled_agents=num_controlled_agents,
                    data_path=eval_dataset,
                    mode='policy',
                    policy=human_policy,
                    select_from_k_scenes=SELECT_FROM,
                    num_episodes=NUM_EVAL_EPISODES,
                    deterministic=DETERMINISTIC,
                )
                
                # Add identifiers
                df_res_bc['Agent'] = bc_config.agent
                df_res_bc['Train agent'] = bc_config.train_agent
                df_res_bc['Dataset'] = dataset
                df_res_bc['Eval mode'] = eval_mode
                df_res_bc['Reg. weight'] = None
                
                # Store
                df_bc = pd.concat([df_bc, df_res_bc], ignore_index=True)

    
    # Concatenate results and store to csv
    df_all = pd.concat([df_ppo_all, df_bc], ignore_index=True)
    df_ppo_all.to_csv(f'df_agg_performance_ip_0131_S{SELECT_FROM}_v2.csv', index=False)
    print(f'Saved!')