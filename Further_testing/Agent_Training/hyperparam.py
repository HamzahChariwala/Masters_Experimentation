import os
import random
import numpy as np
import torch
import yaml
import json
import itertools
import argparse
import sys
import copy
from datetime import datetime
import multiprocessing
from typing import Dict, Any, List

# Add parent directory to sys.path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import functions from train.py
from train import (load_config, create_observation_params, create_policy_kwargs, 
                  set_random_seeds, create_model, create_eval_environments, 
                  create_callbacks, run_training, run_final_evaluation, setup_directories)

import Environment_Tooling.EnvironmentGeneration as Env


class HyperparameterTuner:
    def __init__(self, base_config_path: str, tuning_config_path: str):
        """Initialize the hyperparameter tuner"""
        self.base_config_path = base_config_path
        self.tuning_config_path = tuning_config_path
        with open(base_config_path, 'r') as f:
            self.base_config = yaml.safe_load(f)
        with open(tuning_config_path, 'r') as f:
            self.tuning_config = yaml.safe_load(f)
        self.results = []
    
    def _update_config(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Update base config with specific parameters"""
        updated_config = copy.deepcopy(self.base_config)
        
        for param_path, value in params.items():
            path_parts = param_path.split('.')
            config_section = updated_config
            for part in path_parts[:-1]:
                if part not in config_section:
                    config_section[part] = {}
                config_section = config_section[part]
            config_section[path_parts[-1]] = value
        
        return updated_config
    
    def _evaluate_params(self, params: Dict[str, Any], run_id: str) -> Dict[str, Any]:
        """Train and evaluate an agent with the given parameters"""
        # Setup directories and config
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join("Agent_Storage", "HyperparamTuning", f"run_{run_id}_{timestamp}")
        os.makedirs(output_dir, exist_ok=True)
        
        # MODIFIED: Only update the specified parameters, leave everything else from base config
        config = copy.deepcopy(self.base_config)
        
        # Only apply the specific parameter changes from the tuning config
        for param_path, value in params.items():
            # Split the parameter path (e.g., "model.learning_rate")
            path_parts = param_path.split('.')
            
            # Navigate to the correct part of the config
            config_section = config
            for part in path_parts[:-1]:
                if part not in config_section:
                    config_section[part] = {}
                config_section = config_section[part]
            
            # Update only this specific parameter
            config_section[path_parts[-1]] = value
            
            # Print what we're changing for clarity
            print(f"Tuning parameter: {param_path} = {value}")
        
        # Save config
        with open(os.path.join(output_dir, f"config_{run_id}.yaml"), 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        # Reduce training time
        reduced_factor = self.tuning_config.get('reduced_timesteps_factor', 0.1)
        config['experiment']['output']['total_timesteps'] = int(
            config['experiment']['output']['total_timesteps'] * reduced_factor
        )
        
        # Setup training
        agent_subfolder = os.path.join("HyperparamTuning", f"run_{run_id}_{timestamp}")
        log_dir, performance_log_dir, spawn_vis_dir = setup_directories(config, agent_subfolder)
        observation_params = create_observation_params(config, spawn_vis_dir)
        policy_kwargs = create_policy_kwargs(config)
        set_random_seeds(config)
        
        # Create environment and model
        env = Env.make_parallel_env(
            env_id=config['environment']['id'],
            num_envs=config['environment']['num_envs'],
            env_seed=config['seeds']['environment'],
            use_different_envs=config['environment'].get('use_different_envs', False),
            **observation_params
        )
        model = create_model(config, env, policy_kwargs, log_dir)
        
        # Run training and evaluation
        eval_envs = create_eval_environments(config, observation_params)
        callbacks = create_callbacks(config, eval_envs, performance_log_dir, spawn_vis_dir)
        model = run_training(config, model, callbacks, agent_subfolder)
        rewards, lengths = run_final_evaluation(config, model, observation_params)
        
        # Calculate metrics
        results = {
            "params": params,
            "mean_reward": float(np.mean(rewards)) if rewards else 0,
            "std_reward": float(np.std(rewards)) if rewards else 0,
            "mean_length": float(np.mean(lengths)) if lengths else 0,
            "run_id": run_id,
            "output_dir": output_dir
        }
        
        with open(os.path.join(output_dir, "results.json"), 'w') as f:
            json.dump(results, f, indent=2)
        
        return results
    
    def _save_results(self, search_type: str):
        """Save search results to a file"""
        # Create timestamped directory in HyperparamTuning folder (original location)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = os.path.join("Agent_Storage", "HyperparamTuning", f"results_{timestamp}")
        os.makedirs(results_dir, exist_ok=True)
        
        # Save to the original location
        results_file = os.path.join(results_dir, f"{search_type}_results.json")
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        # Also save to the Hyperparameters folder for easier access
        # Extract the directory from the tuning config path
        config_dir = os.path.dirname(self.tuning_config_path)
        config_results_file = os.path.join(config_dir, f"{search_type}_results.json")
        with open(config_results_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        # Create a human-readable summary file
        summary_file = os.path.join(config_dir, f"{search_type}_summary.txt")
        with open(summary_file, 'w') as f:
            f.write(f"Hyperparameter Tuning Summary - {search_type.replace('_', ' ').title()}\n")
            f.write(f"Run at: {timestamp}\n\n")
            
            # Sort results by mean reward
            sorted_results = sorted(self.results, key=lambda x: x['mean_reward'], reverse=True)
            
            # Write top 10 results or all if fewer
            top_n = min(10, len(sorted_results))
            f.write(f"Top {top_n} configurations:\n")
            f.write("=" * 80 + "\n\n")
            
            for i, result in enumerate(sorted_results[:top_n]):
                f.write(f"Rank {i+1}:\n")
                f.write(f"Mean Reward: {result['mean_reward']:.3f} ± {result['std_reward']:.3f}\n")
                f.write(f"Mean Episode Length: {result['mean_length']:.1f}\n")
                f.write("Parameters:\n")
                for param_name, param_value in result['params'].items():
                    f.write(f"  {param_name}: {param_value}\n")
                f.write(f"Run ID: {result['run_id']}\n")
                f.write(f"Output directory: {result['output_dir']}\n")
                f.write("\n" + "-" * 40 + "\n\n")
            
            # Add recommended configuration
            best = sorted_results[0]
            f.write("RECOMMENDED CONFIGURATION:\n")
            f.write("=" * 80 + "\n\n")
            for param_name, param_value in best['params'].items():
                f.write(f"{param_name}: {param_value}\n")
        
        # Print message about saved files
        print(f"Results saved to {results_file}")
        print(f"Results also copied to {config_results_file}")
        print(f"Human-readable summary saved to {summary_file}")
        
        # Print best results
        self._print_best_results()
    
    def grid_search(self, max_parallel: int = 1) -> List[Dict[str, Any]]:
        """Perform grid search over hyperparameters"""
        param_grid = self.tuning_config.get('grid_search', {})
        param_names = list(param_grid.keys())
        param_values = [param_grid[name] for name in param_names]
        param_combinations = list(itertools.product(*param_values))
        
        print(f"Grid search will evaluate {len(param_combinations)} combinations")
        
        # Create parameter combinations
        tasks = []
        for i, values in enumerate(param_combinations):
            params = {name: value for name, value in zip(param_names, values)}
            run_id = f"grid_{i+1}"
            tasks.append((params, run_id))
        
        # Execute tasks sequentially or in parallel
        if max_parallel <= 1:
            for i, (params, run_id) in enumerate(tasks):
                print(f"\nEvaluating combination {i+1}/{len(tasks)}: {params}")
                self.results.append(self._evaluate_params(params, run_id))
        else:
            with multiprocessing.Pool(processes=min(max_parallel, len(tasks))) as pool:
                self.results = pool.starmap(self._evaluate_params, tasks)
        
        self._save_results("grid_search")
        return self.results
    
    def random_search(self, num_samples: int, max_parallel: int = 1) -> List[Dict[str, Any]]:
        """Perform random search over hyperparameters"""
        param_ranges = self.tuning_config.get('random_search', {})
        
        # Sample parameters
        def sample_param(param_config):
            if isinstance(param_config, list):
                return random.choice(param_config)
            elif isinstance(param_config, dict):
                if param_config.get('distribution') == 'uniform':
                    return random.uniform(param_config['min'], param_config['max'])
                elif param_config.get('distribution') == 'loguniform':
                    log_min, log_max = np.log(param_config['min']), np.log(param_config['max'])
                    return np.exp(random.uniform(log_min, log_max))
                elif param_config.get('distribution') == 'int_uniform':
                    return random.randint(param_config['min'], param_config['max'])
            return param_config
        
        # Generate parameter combinations
        tasks = []
        for i in range(num_samples):
            params = {name: sample_param(config) for name, config in param_ranges.items()}
            tasks.append((params, f"random_{i+1}"))
        
        print(f"Random search will evaluate {num_samples} combinations")
        
        # Execute tasks
        if max_parallel <= 1:
            for i, (params, run_id) in enumerate(tasks):
                print(f"\nEvaluating combination {i+1}/{num_samples}: {params}")
                self.results.append(self._evaluate_params(params, run_id))
        else:
            with multiprocessing.Pool(processes=min(max_parallel, num_samples)) as pool:
                self.results = pool.starmap(self._evaluate_params, tasks)
        
        self._save_results("random_search")
        return self.results
    
    def bayesian_optimization(self, num_trials: int) -> List[Dict[str, Any]]:
        """Perform Bayesian optimization using Optuna"""
        try:
            import optuna
        except ImportError:
            print("Bayesian optimization requires the 'optuna' package.")
            print("Install it with 'pip install optuna'")
            return []
        
        param_space = self.tuning_config.get('bayesian_optimization', {})
        
        # Define the objective function
        def objective(trial):
            params = {}
            for name, config in param_space.items():
                if isinstance(config, list):
                    params[name] = trial.suggest_categorical(name, config)
                elif isinstance(config, dict):
                    if config.get('distribution') == 'uniform':
                        params[name] = trial.suggest_float(name, config['min'], config['max'])
                    elif config.get('distribution') == 'loguniform':
                        params[name] = trial.suggest_float(name, config['min'], config['max'], log=True)
                    elif config.get('distribution') == 'int_uniform':
                        params[name] = trial.suggest_int(name, config['min'], config['max'])
            
            result = self._evaluate_params(params, f"optuna_{trial.number}")
            self.results.append(result)
            return result['mean_reward']
        
        # Create a study and optimize
        print("\nStarting Bayesian optimization with Optuna")
        print(f"Number of trials: {num_trials}")
        print(f"Parameter space: {len(param_space)} parameters")
        for name, config in param_space.items():
            if isinstance(config, list):
                print(f"  {name}: Categorical with {len(config)} options")
            elif isinstance(config, dict):
                if config.get('distribution') == 'uniform':
                    print(f"  {name}: Uniform({config['min']}, {config['max']})")
                elif config.get('distribution') == 'loguniform':
                    print(f"  {name}: LogUniform({config['min']}, {config['max']})")
                elif config.get('distribution') == 'int_uniform':
                    print(f"  {name}: IntUniform({config['min']}, {config['max']})")
        
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=num_trials)
        
        # Save all results to a consolidated file
        self._save_results("bayesian_optimization")
        
        # Save the Optuna study
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = os.path.join("Agent_Storage", "HyperparamTuning", f"results_{timestamp}")
        os.makedirs(results_dir, exist_ok=True)
        
        # Print the best trial from Optuna's perspective
        print("\n====== BEST TRIAL (OPTUNA) ======")
        print(f"Value: {study.best_value:.3f}")
        print("Parameters:")
        for key, value in study.best_params.items():
            print(f"  {key}: {value}")
        
        # Save the complete study object
        with open(os.path.join(results_dir, "optuna_study.pkl"), 'wb') as f:
            import pickle
            pickle.dump(study, f)
        
        # Also save a more readable summary
        with open(os.path.join(results_dir, "optuna_summary.txt"), 'w') as f:
            f.write(f"Optuna Study Summary\n")
            f.write(f"Best trial: #{study.best_trial.number}\n")
            f.write(f"Best value: {study.best_value:.6f}\n\n")
            f.write("Best parameters:\n")
            for key, value in study.best_params.items():
                f.write(f"  {key}: {value}\n")
            
            f.write("\nAll trials:\n")
            for trial in study.trials:
                f.write(f"Trial #{trial.number}: value={trial.value:.6f}\n")
                for key, value in trial.params.items():
                    f.write(f"  {key}: {value}\n")
                f.write("\n")
        
        return self.results
    
    def _print_best_results(self, top_n: int = 5):
        """Print the top N best parameter combinations"""
        if not self.results:
            print("No results to display")
            return
        
        sorted_results = sorted(self.results, key=lambda x: x['mean_reward'], reverse=True)
        
        print("\n====== TOP HYPERPARAMETER COMBINATIONS ======")
        for i, result in enumerate(sorted_results[:top_n]):
            print(f"\nRank {i+1}:")
            print(f"Mean Reward: {result['mean_reward']:.3f} ± {result['std_reward']:.3f}")
            print(f"Mean Episode Length: {result['mean_length']:.1f}")
            print(f"Parameters:")
            for param_name, param_value in result['params'].items():
                print(f"  {param_name}: {param_value}")
            print(f"Output directory: {result['output_dir']}")
        
        # Show best combination
        best = sorted_results[0]
        print("\n====== RECOMMENDED HYPERPARAMETERS ======")
        print("The following combination achieved the best results:")
        for param_name, param_value in best['params'].items():
            print(f"{param_name}: {param_value}")


def main():
    parser = argparse.ArgumentParser(description='Hyperparameter tuning for agent training')
    parser.add_argument('--base-config', type=str, required=True, 
                        help='Path to base configuration YAML file (e.g., Agent_Storage/Hyperparameters/example_config.yaml)')
    parser.add_argument('--tuning-config', type=str, required=True, 
                        help='Path to tuning configuration YAML file with parameters to optimize')
    parser.add_argument('--method', type=str, choices=['grid', 'random', 'bayesian'], 
                        default='grid', help='Search method (default: grid)')
    parser.add_argument('--samples', type=int, default=10, 
                        help='Number of samples for random/bayesian search (default: 10)')
    parser.add_argument('--parallel', type=int, default=1, 
                        help='Number of parallel runs (default: 1)')
    
    args = parser.parse_args()
    
    print(f"\n====== HYPERPARAMETER TUNING ======")
    print(f"Base config: {args.base_config}")
    print(f"Tuning config: {args.tuning_config}")
    print(f"Method: {args.method}")
    print(f"Samples: {args.samples}")
    print(f"Parallel runs: {args.parallel}")
    print(f"====================================\n")
    
    tuner = HyperparameterTuner(args.base_config, args.tuning_config)
    
    if args.method == 'grid':
        results = tuner.grid_search(max_parallel=args.parallel)
    elif args.method == 'random':
        results = tuner.random_search(args.samples, max_parallel=args.parallel)
    elif args.method == 'bayesian':
        results = tuner.bayesian_optimization(args.samples)
    
    print(f"\nCompleted {args.method} search with {len(results)} combinations evaluated")


if __name__ == "__main__":
    main() 