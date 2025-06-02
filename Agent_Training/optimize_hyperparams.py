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
    parser.add_argument('--continue-from', type=str,
                      help='Continue optimization from a previous study directory')
    parser.add_argument('--study-name', type=str,
                      help='Optuna study name (used with --continue-from)')
    parser.add_argument('--enqueue-best', type=int, default=5,
                      help='Number of best trials to enqueue from previous study (0 to disable)')
    
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
        
    # Variables to track previous study info
    previous_results_file = None
    
    # Set up to continue from a previous study if specified
    if args.continue_from:
        previous_study_dir = args.continue_from
        
        # If a direct storage file path is provided
        if previous_study_dir.endswith('.db'):
            tuning_config.storage_name = f"sqlite:///{previous_study_dir}"
            print(f"Continuing from database: {tuning_config.storage_name}")
            
            # Try to find results file in the same directory
            db_dir = os.path.dirname(previous_study_dir)
            possible_results = os.path.join(db_dir, "optimization_results.json")
            if os.path.exists(possible_results):
                previous_results_file = possible_results
        else:
            # Look for the optimization results file to get storage location
            previous_results_file = os.path.join(previous_study_dir, "optimization_results.json")
            if os.path.exists(previous_results_file):
                print(f"Found previous study results at: {previous_results_file}")
                
                # Copy the database if it exists
                db_files = [f for f in os.listdir(previous_study_dir) if f.endswith('.db')]
                if db_files:
                    db_path = os.path.join(previous_study_dir, db_files[0])
                    db_target = os.path.join(output_dir, db_files[0])
                    shutil.copy2(db_path, db_target)
                    tuning_config.storage_name = f"sqlite:///{db_target}"
                    print(f"Copied and using database: {db_target}")
            else:
                print(f"Warning: Could not find results in {previous_study_dir}")
                
        # Use the provided study name if given
        if args.study_name:
            tuning_config.study_name = args.study_name
            print(f"Using study name: {tuning_config.study_name}")
    
    # Set output directory in config
    tuning_config.base_output_dir = output_dir
    
    # Create and run optimizer
    optimizer = OptunaOptimizer(BASE_CONFIG_PATH, tuning_config)
    
    # Load best trials from previous study if requested
    if previous_results_file and args.enqueue_best > 0:
        print(f"Enqueueing top {args.enqueue_best} trials from previous study")
        optimizer.load_best_trials_from_results(previous_results_file, args.enqueue_best)
    
    # Run optimization
    optimizer.optimize()

if __name__ == '__main__':
    main() 