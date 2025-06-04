#!/usr/bin/env python3
"""
Comprehensive Optuna Study Analyzer
Generates multiple visualization plots for hyperparameter optimization analysis.
Uses optimization_results.json and optuna_config.yaml to properly reconstruct the study.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import yaml
from typing import Dict, Any, List, Tuple, Optional
from matplotlib.cm import viridis
from scipy.stats import pearsonr
import optuna
from optuna.visualization import matplotlib as optuna_viz
import plotly.io as pio


class OptunaStudyAnalyzer:
    """Analyzer for Optuna hyperparameter optimization studies."""
    
    def __init__(self, results_file_path: str, config_file_path: str = None, output_dir: str = None):
        """
        Initialize the analyzer with study results.
        
        Args:
            results_file_path: Path to optimization_results.json
            config_file_path: Path to optuna_config.yaml (optional, will auto-detect)
            output_dir: Directory to save plots (default: same as results file)
        """
        self.results_file_path = results_file_path
        self.output_dir = output_dir or os.path.dirname(results_file_path)
        
        # Auto-detect config file if not provided
        if config_file_path is None:
            config_file_path = os.path.join(os.path.dirname(results_file_path), "optuna_config.yaml")
        self.config_file_path = config_file_path
        
        # Load results and config
        self.results = self._load_results()
        self.config = self._load_config()
        
        # Create proper Optuna study
        self.study = self._create_study_from_results()
        
        # Create output directory for plots
        self.plots_dir = os.path.join(self.output_dir, "optuna_plots")
        os.makedirs(self.plots_dir, exist_ok=True)
        
        print(f"Loaded study with {len(self.results['all_trials'])} trials")
        print(f"Pareto front contains {len(self.results['pareto_front'])} trials")
        print(f"Plots will be saved to: {self.plots_dir}")
    
    def _load_results(self) -> Dict[str, Any]:
        """Load results from JSON file."""
        with open(self.results_file_path, 'r') as f:
            return json.load(f)
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        if os.path.exists(self.config_file_path):
            with open(self.config_file_path, 'r') as f:
                return yaml.safe_load(f)
        else:
            print(f"Config file not found: {self.config_file_path}")
            return {}
    
    def _create_distributions_from_config(self) -> Dict[str, optuna.distributions.BaseDistribution]:
        """Create Optuna distributions from config."""
        distributions = {}
        
        if 'bayesian_optimization' in self.config:
            for param_name, param_config in self.config['bayesian_optimization'].items():
                if param_config['distribution'] == 'int_uniform':
                    distributions[param_name] = optuna.distributions.IntDistribution(
                        low=param_config['min'], 
                        high=param_config['max']
                    )
                elif param_config['distribution'] == 'uniform':
                    distributions[param_name] = optuna.distributions.FloatDistribution(
                        low=param_config['min'], 
                        high=param_config['max']
                    )
                elif param_config['distribution'] == 'loguniform':
                    distributions[param_name] = optuna.distributions.FloatDistribution(
                        low=param_config['min'], 
                        high=param_config['max'],
                        log=True
                    )
        
        return distributions
    
    def _create_study_from_results(self) -> optuna.Study:
        """Create an Optuna study object from the results data."""
        # Create a study with the same objectives
        study = optuna.create_study(
            study_name=self.results['study_name'],
            directions=['maximize', 'minimize'],  # goal_reached ↑, lava_proportion ↓
        )
        
        # Get distributions from config
        distributions = self._create_distributions_from_config()
        
        # Add completed trials to the study
        for trial_data in self.results['all_trials']:
            if trial_data['state'] == 'COMPLETE' and trial_data['values'] is not None:
                # Handle both list and individual value formats
                values = trial_data['values']
                if isinstance(values, list) and len(values) == 2:
                    goal_value, lava_value = values
                else:
                    continue  # Skip malformed trials
                
                # Create a frozen trial with proper distributions
                trial = optuna.trial.create_trial(
                    params=trial_data['params'],
                    distributions=distributions,
                    values=[goal_value, lava_value]
                )
                study.add_trial(trial)
        
        print(f"Reconstructed study with {len(study.trials)} trials")
        return study
    
    def generate_all_plots(self) -> List[str]:
        """Generate all available plots including native Optuna visualizations."""
        plot_functions = [
            # Native Optuna visualizations
            self.plot_optuna_pareto_front,
            self.plot_optuna_parameter_importances,
            self.plot_optuna_contour,
            self.plot_optuna_slice,
            self.plot_optuna_parallel_coordinate,
            self.plot_optuna_edf,
            self.plot_optuna_optimization_history,
            
            # Custom matplotlib plots
            self.plot_optimization_history,
            self.plot_parameter_relationships,
            self.plot_param_distributions,
            self.plot_objective_correlation,
            self.plot_trial_timeline,
            self.plot_pareto_analysis,
            self.plot_parameter_sensitivity
        ]
        
        saved_plots = []
        for plot_func in plot_functions:
            try:
                saved_path = plot_func()
                if saved_path:
                    saved_plots.append(saved_path)
                    print(f"✓ Generated: {os.path.basename(saved_path)}")
            except Exception as e:
                print(f"✗ Failed to generate {plot_func.__name__}: {e}")
        
        return saved_plots
    
    def plot_optuna_pareto_front(self) -> str:
        """Plot Pareto front using Optuna's native visualization."""
        try:
            import optuna.visualization as optuna_viz_plotly
            fig = optuna_viz_plotly.plot_pareto_front(self.study)
            fig.update_layout(
                title="Optuna Pareto Front: Goal Reached vs Lava Proportion",
                xaxis_title="Goal Reached Proportion",
                yaxis_title="Lava Proportion"
            )
            
            save_path = os.path.join(self.plots_dir, "optuna_pareto_front.png")
            pio.write_image(fig, save_path)
            return save_path
        except Exception as e:
            print(f"Optuna Pareto front failed: {e}")
            return None
    
    def plot_optuna_parameter_importances(self) -> str:
        """Plot parameter importances using Optuna's native analysis."""
        try:
            import optuna.visualization as optuna_viz_plotly
            
            # Create plots for both objectives
            fig_goal = optuna_viz_plotly.plot_param_importances(
                self.study, target=lambda t: t.values[0]
            )
            fig_goal.update_layout(title="Parameter Importance: Goal Reached")
            
            fig_lava = optuna_viz_plotly.plot_param_importances(
                self.study, target=lambda t: t.values[1]
            )
            fig_lava.update_layout(title="Parameter Importance: Lava Proportion")
            
            # Save both plots
            goal_path = os.path.join(self.plots_dir, "optuna_param_importance_goal.png")
            lava_path = os.path.join(self.plots_dir, "optuna_param_importance_lava.png")
            
            pio.write_image(fig_goal, goal_path)
            pio.write_image(fig_lava, lava_path)
            
            return goal_path  # Return one path for counting
        except Exception as e:
            print(f"Optuna parameter importance failed: {e}")
            return None
    
    def plot_optuna_contour(self) -> str:
        """Plot contour maps for parameter pairs using Optuna's native visualization."""
        try:
            import optuna.visualization as optuna_viz_plotly
            
            # Create contour plots for both objectives
            fig_goal = optuna_viz_plotly.plot_contour(
                self.study, target=lambda t: t.values[0]
            )
            fig_goal.update_layout(title="Parameter Contour Maps - Goal Reached")
            
            fig_lava = optuna_viz_plotly.plot_contour(
                self.study, target=lambda t: t.values[1]
            )
            fig_lava.update_layout(title="Parameter Contour Maps - Lava Proportion")
            
            # Save both plots
            goal_path = os.path.join(self.plots_dir, "optuna_contour_goal.png")
            lava_path = os.path.join(self.plots_dir, "optuna_contour_lava.png")
            
            pio.write_image(fig_goal, goal_path)
            pio.write_image(fig_lava, lava_path)
            
            return goal_path  # Return one path for counting
        except Exception as e:
            print(f"Optuna contour plot failed: {e}")
            return None
    
    def plot_optuna_slice(self) -> str:
        """Plot parameter slice plots using Optuna's native visualization."""
        try:
            import optuna.visualization as optuna_viz_plotly
            
            # Create slice plots for both objectives
            fig_goal = optuna_viz_plotly.plot_slice(
                self.study, target=lambda t: t.values[0]
            )
            fig_goal.update_layout(title="Parameter Slice Plots - Goal Reached")
            
            fig_lava = optuna_viz_plotly.plot_slice(
                self.study, target=lambda t: t.values[1]
            )
            fig_lava.update_layout(title="Parameter Slice Plots - Lava Proportion")
            
            # Save both plots
            goal_path = os.path.join(self.plots_dir, "optuna_slice_goal.png")
            lava_path = os.path.join(self.plots_dir, "optuna_slice_lava.png")
            
            pio.write_image(fig_goal, goal_path)
            pio.write_image(fig_lava, lava_path)
            
            return goal_path  # Return one path for counting
        except Exception as e:
            print(f"Optuna slice plot failed: {e}")
            return None
    
    def plot_optuna_parallel_coordinate(self) -> str:
        """Plot parallel coordinate plot using Optuna's native visualization."""
        try:
            import optuna.visualization as optuna_viz_plotly
            
            # Create parallel coordinate plots for both objectives
            fig_goal = optuna_viz_plotly.plot_parallel_coordinate(
                self.study, target=lambda t: t.values[0]
            )
            fig_goal.update_layout(title="Parallel Coordinate Plot - Goal Reached")
            
            fig_lava = optuna_viz_plotly.plot_parallel_coordinate(
                self.study, target=lambda t: t.values[1]
            )
            fig_lava.update_layout(title="Parallel Coordinate Plot - Lava Proportion")
            
            # Save both plots
            goal_path = os.path.join(self.plots_dir, "optuna_parallel_goal.png")
            lava_path = os.path.join(self.plots_dir, "optuna_parallel_lava.png")
            
            pio.write_image(fig_goal, goal_path)
            pio.write_image(fig_lava, lava_path)
            
            return goal_path  # Return one path for counting
        except Exception as e:
            print(f"Optuna parallel coordinate plot failed: {e}")
            return None
    
    def plot_optuna_edf(self) -> str:
        """Plot empirical distribution function using Optuna's native visualization."""
        try:
            import optuna.visualization as optuna_viz_plotly
            
            # Create EDF plots for both objectives
            fig_goal = optuna_viz_plotly.plot_edf(
                self.study, target=lambda t: t.values[0]
            )
            fig_goal.update_layout(title="Empirical Distribution Function - Goal Reached")
            
            fig_lava = optuna_viz_plotly.plot_edf(
                self.study, target=lambda t: t.values[1]
            )
            fig_lava.update_layout(title="Empirical Distribution Function - Lava Proportion")
            
            # Save both plots
            goal_path = os.path.join(self.plots_dir, "optuna_edf_goal.png")
            lava_path = os.path.join(self.plots_dir, "optuna_edf_lava.png")
            
            pio.write_image(fig_goal, goal_path)
            pio.write_image(fig_lava, lava_path)
            
            return goal_path  # Return one path for counting
        except Exception as e:
            print(f"Optuna EDF plot failed: {e}")
            return None
    
    def plot_optuna_optimization_history(self) -> str:
        """Plot optimization history using Optuna's native visualization."""
        try:
            import optuna.visualization as optuna_viz_plotly
            
            # Create optimization history plots for both objectives
            fig_goal = optuna_viz_plotly.plot_optimization_history(
                self.study, target=lambda t: t.values[0]
            )
            fig_goal.update_layout(title="Optuna Optimization History - Goal Reached")
            
            fig_lava = optuna_viz_plotly.plot_optimization_history(
                self.study, target=lambda t: t.values[1]
            )
            fig_lava.update_layout(title="Optuna Optimization History - Lava Proportion")
            
            # Save both plots
            goal_path = os.path.join(self.plots_dir, "optuna_history_goal.png")
            lava_path = os.path.join(self.plots_dir, "optuna_history_lava.png")
            
            pio.write_image(fig_goal, goal_path)
            pio.write_image(fig_lava, lava_path)
            
            return goal_path  # Return one path for counting
        except Exception as e:
            print(f"Optuna optimization history failed: {e}")
            return None
    
    def plot_pareto_front(self) -> str:
        """Plot Pareto front for multi-objective optimization."""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Get all trial data
        all_trials = []
        pareto_trials = []
        
        for trial_data in self.results['all_trials']:
            if trial_data['state'] == 'COMPLETE' and trial_data['values'] is not None:
                goal_reached, lava_prop = trial_data['values']
                all_trials.append((goal_reached, lava_prop))
        
        for trial_data in self.results['pareto_front']:
            goal_reached, lava_prop = trial_data['values']
            pareto_trials.append((goal_reached, lava_prop))
        
        # Plot all trials
        if all_trials:
            goals, lavas = zip(*all_trials)
            ax.scatter(goals, lavas, alpha=0.6, c='lightblue', s=30, label='All Trials')
        
        # Plot Pareto front
        if pareto_trials:
            p_goals, p_lavas = zip(*pareto_trials)
            ax.scatter(p_goals, p_lavas, c='red', s=100, label='Pareto Optimal', 
                      edgecolors='black', linewidth=1)
        
        ax.set_xlabel('Goal Reached Proportion')
        ax.set_ylabel('Lava Proportion')
        ax.set_title('Pareto Front: Goal Reached vs Lava Proportion')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        save_path = os.path.join(self.plots_dir, "pareto_front_matplotlib.png")
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        return save_path
    
    def plot_optimization_history(self) -> str:
        """Plot optimization history over trials."""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Extract trial data
        trials = []
        for trial_data in self.results['all_trials']:
            if trial_data['state'] == 'COMPLETE' and trial_data['values'] is not None:
                trials.append({
                    'number': trial_data['number'],
                    'goal_reached': trial_data['values'][0],
                    'lava_proportion': trial_data['values'][1]
                })
        
        if not trials:
            return None
        
        trials.sort(key=lambda x: x['number'])
        trial_numbers = [t['number'] for t in trials]
        goal_values = [t['goal_reached'] for t in trials]
        lava_values = [t['lava_proportion'] for t in trials]
        
        # Plot goal reached history
        ax1.plot(trial_numbers, goal_values, 'b-', alpha=0.7, linewidth=1)
        ax1.scatter(trial_numbers, goal_values, c='blue', s=20, alpha=0.6)
        ax1.set_ylabel('Goal Reached Proportion')
        ax1.set_title('Optimization History: Goal Reached Proportion')
        ax1.grid(True, alpha=0.3)
        
        # Add running maximum
        running_max = np.maximum.accumulate(goal_values)
        ax1.plot(trial_numbers, running_max, 'r-', linewidth=2, label='Running Maximum')
        ax1.legend()
        
        # Plot lava proportion history
        ax2.plot(trial_numbers, lava_values, 'g-', alpha=0.7, linewidth=1)
        ax2.scatter(trial_numbers, lava_values, c='green', s=20, alpha=0.6)
        ax2.set_xlabel('Trial Number')
        ax2.set_ylabel('Lava Proportion')
        ax2.set_title('Optimization History: Lava Proportion')
        ax2.grid(True, alpha=0.3)
        
        # Add running minimum
        running_min = np.minimum.accumulate(lava_values)
        ax2.plot(trial_numbers, running_min, 'r-', linewidth=2, label='Running Minimum')
        ax2.legend()
        
        save_path = os.path.join(self.plots_dir, "optimization_history_matplotlib.png")
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        return save_path
    
    def plot_parameter_relationships(self) -> str:
        """Plot relationships between parameters and objectives."""
        # Get all parameter names
        param_names = set()
        for trial_data in self.results['all_trials']:
            if trial_data['state'] == 'COMPLETE':
                param_names.update(trial_data['params'].keys())
        
        param_names = sorted(list(param_names))
        n_params = len(param_names)
        
        if n_params == 0:
            return None
        
        # Create subplots for first 4 parameters
        n_to_plot = min(4, n_params)
        fig, axes = plt.subplots(2, n_to_plot, figsize=(4*n_to_plot, 10))
        if n_to_plot == 1:
            axes = axes.reshape(2, 1)
        
        # Plot first 4 parameters
        for i, param_name in enumerate(param_names[:n_to_plot]):
            param_values = []
            goal_values = []
            lava_values = []
            
            for trial_data in self.results['all_trials']:
                if (trial_data['state'] == 'COMPLETE' and 
                    trial_data['values'] is not None and 
                    param_name in trial_data['params']):
                    param_values.append(trial_data['params'][param_name])
                    goal_values.append(trial_data['values'][0])
                    lava_values.append(trial_data['values'][1])
            
            if param_values:
                # Goal reached vs parameter
                axes[0, i].scatter(param_values, goal_values, alpha=0.6, s=30)
                axes[0, i].set_xlabel(param_name.replace('model.', ''))
                axes[0, i].set_ylabel('Goal Reached')
                axes[0, i].set_title(f'Goal vs {param_name.replace("model.", "")}')
                axes[0, i].grid(True, alpha=0.3)
                
                # Lava proportion vs parameter
                axes[1, i].scatter(param_values, lava_values, alpha=0.6, s=30, c='green')
                axes[1, i].set_xlabel(param_name.replace('model.', ''))
                axes[1, i].set_ylabel('Lava Proportion')
                axes[1, i].set_title(f'Lava vs {param_name.replace("model.", "")}')
                axes[1, i].grid(True, alpha=0.3)
        
        save_path = os.path.join(self.plots_dir, "parameter_relationships_matplotlib.png")
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        return save_path
    
    def plot_param_distributions(self) -> str:
        """Plot parameter value distributions."""
        # Get all parameter names
        param_names = set()
        for trial_data in self.results['all_trials']:
            if trial_data['state'] == 'COMPLETE':
                param_names.update(trial_data['params'].keys())
        
        param_names = sorted(list(param_names))
        n_params = len(param_names)
        
        if n_params == 0:
            return None
        
        # Create subplots
        rows = (n_params + 2) // 3
        cols = min(3, n_params)
        fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
        
        if rows == 1 and cols == 1:
            axes = [axes]
        elif rows == 1:
            axes = axes.reshape(1, -1)
        elif cols == 1:
            axes = axes.reshape(-1, 1)
        
        for i, param_name in enumerate(param_names):
            row = i // cols
            col = i % cols
            
            param_values = []
            for trial_data in self.results['all_trials']:
                if (trial_data['state'] == 'COMPLETE' and 
                    param_name in trial_data['params']):
                    param_values.append(trial_data['params'][param_name])
            
            if param_values:
                if rows == 1 and cols == 1:
                    ax = axes[0]
                elif rows == 1:
                    ax = axes[0, col]
                elif cols == 1:
                    ax = axes[row, 0]
                else:
                    ax = axes[row, col]
                
                ax.hist(param_values, bins=20, alpha=0.7, edgecolor='black')
                ax.set_xlabel(param_name.replace('model.', ''))
                ax.set_ylabel('Frequency')
                ax.set_title(f'Distribution: {param_name.replace("model.", "")}')
                ax.grid(True, alpha=0.3)
        
        # Hide empty subplots
        total_subplots = rows * cols
        for i in range(n_params, total_subplots):
            row = i // cols
            col = i % cols
            if rows == 1:
                axes[0, col].set_visible(False)
            elif cols == 1:
                axes[row, 0].set_visible(False)
            else:
                axes[row, col].set_visible(False)
        
        save_path = os.path.join(self.plots_dir, "parameter_distributions_matplotlib.png")
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        return save_path
    
    def plot_objective_correlation(self) -> str:
        """Plot correlation between objectives."""
        goal_values = []
        lava_values = []
        
        for trial_data in self.results['all_trials']:
            if trial_data['state'] == 'COMPLETE' and trial_data['values'] is not None:
                goal_values.append(trial_data['values'][0])
                lava_values.append(trial_data['values'][1])
        
        if len(goal_values) < 2:
            return None
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Scatter plot
        ax.scatter(goal_values, lava_values, alpha=0.6, s=50)
        
        # Calculate and display correlation
        correlation = np.corrcoef(goal_values, lava_values)[0, 1]
        
        ax.set_xlabel('Goal Reached Proportion')
        ax.set_ylabel('Lava Proportion')
        ax.set_title(f'Objective Correlation (r = {correlation:.3f})')
        ax.grid(True, alpha=0.3)
        
        # Add trend line
        z = np.polyfit(goal_values, lava_values, 1)
        p = np.poly1d(z)
        ax.plot(sorted(goal_values), p(sorted(goal_values)), "r--", alpha=0.8)
        
        save_path = os.path.join(self.plots_dir, "objective_correlation_matplotlib.png")
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        return save_path
    
    def plot_trial_timeline(self) -> str:
        """Plot trial completion timeline."""
        completed_trials = []
        for trial_data in self.results['all_trials']:
            if trial_data['state'] == 'COMPLETE':
                completed_trials.append(trial_data['number'])
        
        if not completed_trials:
            return None
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot cumulative completed trials
        completed_trials.sort()
        cumulative = list(range(1, len(completed_trials) + 1))
        
        ax.plot(completed_trials, cumulative, 'b-', linewidth=2, marker='o', markersize=3)
        ax.set_xlabel('Trial Number')
        ax.set_ylabel('Cumulative Completed Trials')
        ax.set_title('Trial Completion Timeline')
        ax.grid(True, alpha=0.3)
        
        save_path = os.path.join(self.plots_dir, "trial_timeline_matplotlib.png")
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        return save_path
    
    def plot_pareto_analysis(self) -> str:
        """Detailed analysis of Pareto optimal trials."""
        pareto_trials = self.results['pareto_front']
        if not pareto_trials:
            return None
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Extract data
        goal_values = [trial['values'][0] for trial in pareto_trials]
        lava_values = [trial['values'][1] for trial in pareto_trials]
        
        # 1. Pareto front with trial numbers
        ax1.scatter(goal_values, lava_values, c='red', s=100, edgecolors='black')
        for trial in pareto_trials:
            ax1.annotate(f"T{trial['number']}", 
                        (trial['values'][0], trial['values'][1]),
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        ax1.set_xlabel('Goal Reached Proportion')
        ax1.set_ylabel('Lava Proportion')
        ax1.set_title('Pareto Optimal Trials with Numbers')
        ax1.grid(True, alpha=0.3)
        
        # 2. Goal vs Lava trade-off
        sorted_indices = np.argsort(goal_values)
        sorted_goals = np.array(goal_values)[sorted_indices]
        sorted_lavas = np.array(lava_values)[sorted_indices]
        
        ax2.plot(sorted_goals, sorted_lavas, 'ro-', linewidth=2, markersize=8)
        ax2.set_xlabel('Goal Reached Proportion')
        ax2.set_ylabel('Lava Proportion')
        ax2.set_title('Pareto Trade-off Curve')
        ax2.grid(True, alpha=0.3)
        
        # 3. Parameter ranges for Pareto trials
        param_names = list(pareto_trials[0]['params'].keys())[:4]  # First 4 params
        param_ranges = []
        
        for param_name in param_names:
            values = [trial['params'][param_name] for trial in pareto_trials]
            param_ranges.append((param_name.replace('model.', ''), min(values), max(values)))
        
        param_labels, min_vals, max_vals = zip(*param_ranges)
        y_pos = np.arange(len(param_labels))
        
        ax3.barh(y_pos, [max_val - min_val for min_val, max_val in zip(min_vals, max_vals)],
                left=min_vals, alpha=0.7)
        ax3.set_yticks(y_pos)
        ax3.set_yticklabels(param_labels)
        ax3.set_xlabel('Parameter Value Range')
        ax3.set_title('Parameter Ranges in Pareto Optimal Trials')
        ax3.grid(True, alpha=0.3)
        
        # 4. Performance distribution
        ax4.hist(goal_values, bins=5, alpha=0.7, label='Goal Reached', color='blue')
        ax4.hist(lava_values, bins=5, alpha=0.7, label='Lava Proportion', color='orange')
        ax4.set_xlabel('Value')
        ax4.set_ylabel('Frequency')
        ax4.set_title('Performance Distribution in Pareto Set')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        save_path = os.path.join(self.plots_dir, "pareto_analysis_matplotlib.png")
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        return save_path
    
    def plot_parameter_sensitivity(self) -> str:
        """Analyze parameter sensitivity to objectives."""
        # Get all parameter names
        param_names = set()
        for trial_data in self.results['all_trials']:
            if trial_data['state'] == 'COMPLETE':
                param_names.update(trial_data['params'].keys())
        
        param_names = sorted(list(param_names))
        
        if len(param_names) == 0:
            return None
        
        # Calculate correlations
        goal_correlations = []
        lava_correlations = []
        
        for param_name in param_names:
            param_values = []
            goal_values = []
            lava_values = []
            
            for trial_data in self.results['all_trials']:
                if (trial_data['state'] == 'COMPLETE' and 
                    trial_data['values'] is not None and 
                    param_name in trial_data['params']):
                    param_values.append(trial_data['params'][param_name])
                    goal_values.append(trial_data['values'][0])
                    lava_values.append(trial_data['values'][1])
            
            if len(param_values) > 1:
                goal_corr, _ = pearsonr(param_values, goal_values)
                lava_corr, _ = pearsonr(param_values, lava_values)
                goal_correlations.append(abs(goal_corr))
                lava_correlations.append(abs(lava_corr))
            else:
                goal_correlations.append(0)
                lava_correlations.append(0)
        
        # Create plots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Goal reached correlations
        param_labels = [name.replace('model.', '') for name in param_names]
        y_pos = np.arange(len(param_labels))
        
        ax1.barh(y_pos, goal_correlations, alpha=0.7, color='blue')
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels(param_labels)
        ax1.set_xlabel('Absolute Correlation')
        ax1.set_title('Parameter Sensitivity: Goal Reached')
        ax1.grid(True, alpha=0.3)
        
        # Lava proportion correlations
        ax2.barh(y_pos, lava_correlations, alpha=0.7, color='orange')
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(param_labels)
        ax2.set_xlabel('Absolute Correlation')
        ax2.set_title('Parameter Sensitivity: Lava Proportion')
        ax2.grid(True, alpha=0.3)
        
        save_path = os.path.join(self.plots_dir, "parameter_sensitivity_matplotlib.png")
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        return save_path


def main():
    """Main function to run the Optuna study analysis."""
    print("Starting comprehensive Optuna study analysis...")
    
    # Default path to the optimization results (relative to project root)
    results_file = "../../Agent_Storage/Hyperparameters/optimization_20250529_010111/optimization_results.json"
    
    print(f"Looking for results file: {results_file}")
    
    if not os.path.exists(results_file):
        print(f"Results file not found: {results_file}")
        print("Please provide the correct path to optimization_results.json")
        return
    
    print("Results file found! Creating analyzer with native Optuna visualizations...")
    
    # Create analyzer
    analyzer = OptunaStudyAnalyzer(results_file)
    
    # Generate all plots
    print("\nGenerating comprehensive Optuna study visualization plots...")
    print("This includes native Optuna visualizations: contour maps, parameter importance, slice plots, etc.")
    saved_plots = analyzer.generate_all_plots()
    
    print(f"\nAnalysis complete! Generated {len(saved_plots)} plots.")
    print(f"All plots saved to: {analyzer.plots_dir}")
    
    # Print summary
    print(f"\nStudy Summary:")
    print(f"- Total trials: {len(analyzer.results['all_trials'])}")
    print(f"- Pareto optimal trials: {len(analyzer.results['pareto_front'])}")
    print(f"- Study name: {analyzer.results['study_name']}")
    
    # Print available Optuna features accessed
    print(f"\nNative Optuna Features Accessed:")
    print("- Parameter importance analysis")
    print("- Contour maps for parameter pairs")
    print("- Slice plots for individual parameters")
    print("- Parallel coordinate plots")
    print("- Empirical distribution functions")
    print("- Native Pareto front visualization")
    print("- Native optimization history")


if __name__ == "__main__":
    main() 