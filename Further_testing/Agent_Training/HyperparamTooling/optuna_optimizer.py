"""Main optimization module using Optuna for multi-objective hyperparameter tuning."""

import os
import json
import optuna
import yaml
from datetime import datetime
from typing import Dict, Any, List, Tuple
import logging

from .tuning_config import TuningConfig
from .trial_evaluation import create_trial_directory, evaluate_trial

class OptunaOptimizer:
    def __init__(self, base_config_path: str, tuning_config: TuningConfig = None):
        """Initialize the Optuna optimizer."""
        # Load configurations
        with open(base_config_path, 'r') as f:
            self.base_config = yaml.safe_load(f)
        
        self.tuning_config = tuning_config or TuningConfig()
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Create output directory
        os.makedirs(self.tuning_config.base_output_dir, exist_ok=True)
        
        # Save configurations
        self._save_configs()
        
        # Setup study
        self.study = optuna.create_study(
            study_name=self.tuning_config.study_name,
            storage=self.tuning_config.storage_name,
            directions=['maximize', 'minimize'],  # goal_reached ↑, lava_proportion ↓
            load_if_exists=True
        )
    
    def _save_configs(self):
        """Save configurations to the output directory."""
        # Save base config
        base_config_path = os.path.join(self.tuning_config.base_output_dir, "base_config.yaml")
        with open(base_config_path, 'w') as f:
            yaml.dump(self.base_config, f, default_flow_style=False)
        
        # Save tuning config
        tuning_config_path = os.path.join(self.tuning_config.base_output_dir, "tuning_config.yaml")
        self.tuning_config.to_yaml(tuning_config_path)
    
    def _suggest_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Suggest parameters for a trial."""
        params = {}
        for param_name, param_config in self.tuning_config.bayesian_optimization.items():
            if param_config['distribution'] == 'int_uniform':
                params[param_name] = trial.suggest_int(
                    param_name, 
                    param_config['min'], 
                    param_config['max']
                )
            elif param_config['distribution'] == 'uniform':
                params[param_name] = trial.suggest_float(
                    param_name, 
                    param_config['min'], 
                    param_config['max']
                )
            elif param_config['distribution'] == 'loguniform':
                params[param_name] = trial.suggest_float(
                    param_name, 
                    param_config['min'], 
                    param_config['max'],
                    log=True
                )
        return params
    
    def objective(self, trial: optuna.Trial) -> Tuple[float, float]:
        """Objective function for Optuna optimization."""
        # Suggest parameters
        params = self._suggest_params(trial)
        
        # Create trial directory
        trial_dir = create_trial_directory(self.tuning_config.base_output_dir, trial.number)
        
        # Log trial start
        self.logger.info(f"\nStarting trial {trial.number}")
        self.logger.info(f"Parameters: {params}")
        
        # Evaluate trial
        goal_proportion, lava_proportion = evaluate_trial(
            self.base_config,
            params,
            trial_dir,
            self.tuning_config.agents_per_trial,
            self.tuning_config.reduced_timesteps_factor
        )
        
        # Log results
        self.logger.info(f"Trial {trial.number} results:")
        self.logger.info(f"Goal reached proportion: {goal_proportion:.3f}")
        self.logger.info(f"Lava proportion: {lava_proportion:.3f}")
        
        return goal_proportion, lava_proportion
    
    def optimize(self) -> None:
        """Run the optimization process."""
        self.logger.info("Starting optimization")
        self.logger.info(f"Output directory: {self.tuning_config.base_output_dir}")
        self.logger.info(f"Number of trials: {self.tuning_config.n_trials}")
        self.logger.info(f"Agents per trial: {self.tuning_config.agents_per_trial}")
        
        # Run optimization
        self.study.optimize(
            self.objective,
            n_trials=self.tuning_config.n_trials,
            show_progress_bar=True
        )
        
        # Save results
        self._save_results()
    
    def _save_results(self) -> None:
        """Save optimization results."""
        # Get trials
        trials = self.study.trials
        
        # Create results summary
        results = {
            'study_name': self.study.study_name,
            'n_trials': len(trials),
            'pareto_front': [
                {
                    'number': trial.number,
                    'params': trial.params,
                    'values': trial.values
                }
                for trial in self.study.best_trials
            ],
            'all_trials': [
                {
                    'number': trial.number,
                    'params': trial.params,
                    'values': trial.values,
                    'state': trial.state.name
                }
                for trial in trials
            ]
        }
        
        # Save results
        results_path = os.path.join(self.tuning_config.base_output_dir, "optimization_results.json")
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Create human-readable summary
        summary_path = os.path.join(self.tuning_config.base_output_dir, "optimization_summary.txt")
        with open(summary_path, 'w') as f:
            f.write("Hyperparameter Optimization Summary\n")
            f.write("=================================\n\n")
            
            f.write(f"Study Name: {self.study.study_name}\n")
            f.write(f"Number of Trials: {len(trials)}\n")
            f.write(f"Number of Pareto Optimal Trials: {len(self.study.best_trials)}\n\n")
            
            f.write("Pareto Optimal Solutions:\n")
            f.write("------------------------\n")
            for trial in self.study.best_trials:
                f.write(f"\nTrial {trial.number}:\n")
                f.write(f"Goal Reached Proportion: {trial.values[0]:.3f}\n")
                f.write(f"Lava Proportion: {trial.values[1]:.3f}\n")
                f.write("Parameters:\n")
                for param_name, param_value in trial.params.items():
                    f.write(f"  {param_name}: {param_value}\n")
        
        self.logger.info(f"Results saved to {self.tuning_config.base_output_dir}")
        self.logger.info(f"Found {len(self.study.best_trials)} Pareto optimal solutions") 