#!/usr/bin/env python3
"""Main script for running hyperparameter optimization."""

import os
import argparse
import shutil
from datetime import datetime
from HyperparamTooling.tuning_config import TuningConfig
from HyperparamTooling.optuna_optimizer import OptunaOptimizer

# Hardcoded paths
BASE_CONFIG_PATH = "Agent_Storage/Hyperparameters/example_config.yaml"
OPTUNA_CONFIG_PATH = "Agent_Storage/Hyperparameters/optuna_config.yaml"

def main():
    parser = argparse.ArgumentParser(description='Run hyperparameter optimization')
    parser.add_argument('--n-trials', type=int, default=2,
                      help='Number of trials to run in the Optuna optimization')
    parser.add_argument('--agents-per-trial', type=int, default=1,
                      help='Number of agents to train per trial')
    parser.add_argument('--storage', type=str,
                      help='Optuna storage string (e.g., sqlite:///study.db)')
    
    args = parser.parse_args()
    
    # Create timestamp for this optimization run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create output directory
    output_dir = os.path.join("Agent_Storage/Hyperparameters", f"optimization_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Copy config files to output directory for auditing
    shutil.copy2(BASE_CONFIG_PATH, os.path.join(output_dir, "base_config.yaml"))
    shutil.copy2(OPTUNA_CONFIG_PATH, os.path.join(output_dir, "optuna_config.yaml"))
    
    # Create tuning config
    tuning_config = TuningConfig.from_yaml(OPTUNA_CONFIG_PATH)
    
    # Update config with command line arguments
    tuning_config.n_trials = args.n_trials
    tuning_config.agents_per_trial = args.agents_per_trial
    if args.storage:
        tuning_config.storage_name = args.storage
    
    # Set output directory in config
    tuning_config.base_output_dir = output_dir
    
    # Create and run optimizer
    optimizer = OptunaOptimizer(BASE_CONFIG_PATH, tuning_config)
    optimizer.optimize()

if __name__ == '__main__':
    main() 