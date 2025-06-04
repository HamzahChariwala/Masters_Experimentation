#!/usr/bin/env python3
"""
Individual Optuna Plot Extractor
Extracts individual contour maps, slice plots, and creates blown-up parallel coordinate plots.
Organizes them in subfolders to avoid cluttering the main directory.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import yaml
import optuna
import plotly.io as pio
import plotly.graph_objects as go
from itertools import combinations
from typing import Dict, Any, List, Tuple


class IndividualOptunaPlotExtractor:
    """Extracts individual plots from Optuna study."""
    
    def __init__(self, results_file_path: str, config_file_path: str = None, output_dir: str = None):
        """Initialize the extractor."""
        self.results_file_path = results_file_path
        self.output_dir = output_dir or "optuna_initial"
        
        # Auto-detect config file
        if config_file_path is None:
            config_file_path = os.path.join(os.path.dirname(results_file_path), "optuna_config.yaml")
        self.config_file_path = config_file_path
        
        # Load data
        self.results = self._load_results()
        self.config = self._load_config()
        self.study = self._create_study_from_results()
        
        # Create subfolders
        self.contour_dir = os.path.join(self.output_dir, "individual_contours")
        self.slice_dir = os.path.join(self.output_dir, "individual_slices")
        self.parallel_dir = os.path.join(self.output_dir, "blown_up_parallel")
        
        for folder in [self.contour_dir, self.slice_dir, self.parallel_dir]:
            os.makedirs(folder, exist_ok=True)
    
    def _load_results(self) -> Dict[str, Any]:
        """Load results from JSON file."""
        with open(self.results_file_path, 'r') as f:
            return json.load(f)
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        if os.path.exists(self.config_file_path):
            with open(self.config_file_path, 'r') as f:
                return yaml.safe_load(f)
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
        study = optuna.create_study(
            study_name=self.results['study_name'],
            directions=['maximize', 'minimize'],
        )
        
        distributions = self._create_distributions_from_config()
        
        for trial_data in self.results['all_trials']:
            if trial_data['state'] == 'COMPLETE' and trial_data['values'] is not None:
                values = trial_data['values']
                if isinstance(values, list) and len(values) == 2:
                    goal_value, lava_value = values
                    trial = optuna.trial.create_trial(
                        params=trial_data['params'],
                        distributions=distributions,
                        values=[goal_value, lava_value]
                    )
                    study.add_trial(trial)
        
        return study
    
    def extract_individual_contour_maps(self):
        """Extract individual contour maps for each parameter pair."""
        print("Extracting individual contour maps...")
        
        # Get parameter names
        param_names = list(self.study.trials[0].params.keys())
        
        # Create contour maps for each parameter pair and each objective
        objectives = [
            ("goal", lambda t: t.values[0], "Goal Reached"),
            ("lava", lambda t: t.values[1], "Lava Proportion")
        ]
        
        for obj_name, obj_func, obj_title in objectives:
            for param1, param2 in combinations(param_names, 2):
                try:
                    import optuna.visualization as optuna_viz_plotly
                    
                    # Create contour plot for this parameter pair
                    fig = optuna_viz_plotly.plot_contour(
                        self.study, 
                        params=[param1, param2],
                        target=obj_func
                    )
                    
                    # Update layout to remove/minimize point size
                    fig.update_traces(
                        marker=dict(size=2, opacity=0.5),  # Much smaller points
                        selector=dict(type='scatter')
                    )
                    
                    # Clean parameter names for title
                    clean_param1 = param1.replace('model.', '')
                    clean_param2 = param2.replace('model.', '')
                    
                    fig.update_layout(
                        title=f"Contour Map: {clean_param1} vs {clean_param2} ({obj_title})",
                        width=800,
                        height=600
                    )
                    
                    # Save individual plot
                    filename = f"contour_{obj_name}_{clean_param1}_vs_{clean_param2}.png"
                    save_path = os.path.join(self.contour_dir, filename)
                    pio.write_image(fig, save_path)
                    
                    print(f"  ✓ Saved: {filename}")
                    
                except Exception as e:
                    print(f"  ✗ Failed contour {param1} vs {param2} ({obj_name}): {e}")
    
    def extract_individual_slice_plots(self):
        """Extract individual slice plots for each parameter."""
        print("\nExtracting individual slice plots...")
        
        param_names = list(self.study.trials[0].params.keys())
        objectives = [
            ("goal", lambda t: t.values[0], "Goal Reached"),
            ("lava", lambda t: t.values[1], "Lava Proportion")
        ]
        
        for obj_name, obj_func, obj_title in objectives:
            for param_name in param_names:
                try:
                    import optuna.visualization as optuna_viz_plotly
                    
                    # Create slice plot for this parameter
                    fig = optuna_viz_plotly.plot_slice(
                        self.study,
                        params=[param_name],
                        target=obj_func
                    )
                    
                    # Update layout
                    clean_param = param_name.replace('model.', '')
                    fig.update_layout(
                        title=f"Slice Plot: {clean_param} vs {obj_title}",
                        width=800,
                        height=600
                    )
                    
                    # Save individual plot
                    filename = f"slice_{obj_name}_{clean_param}.png"
                    save_path = os.path.join(self.slice_dir, filename)
                    pio.write_image(fig, save_path)
                    
                    print(f"  ✓ Saved: {filename}")
                    
                except Exception as e:
                    print(f"  ✗ Failed slice {param_name} ({obj_name}): {e}")
    
    def create_blown_up_parallel_plots(self):
        """Create blown-up versions of parallel coordinate plots."""
        print("\nCreating blown-up parallel coordinate plots...")
        
        objectives = [
            ("goal", lambda t: t.values[0], "Goal Reached"),
            ("lava", lambda t: t.values[1], "Lava Proportion")
        ]
        
        for obj_name, obj_func, obj_title in objectives:
            try:
                import optuna.visualization as optuna_viz_plotly
                
                # Create parallel coordinate plot
                fig = optuna_viz_plotly.plot_parallel_coordinate(
                    self.study,
                    target=obj_func
                )
                
                # Make it much larger and improve readability
                fig.update_layout(
                    title=f"Parallel Coordinate Plot: {obj_title}",
                    width=1400,  # Much wider
                    height=800,  # Much taller
                    font=dict(size=14),  # Larger font
                    title_font=dict(size=18)
                )
                
                # Save blown-up version
                filename = f"parallel_blown_up_{obj_name}.png"
                save_path = os.path.join(self.parallel_dir, filename)
                pio.write_image(fig, save_path)
                
                print(f"  ✓ Saved: {filename}")
                
            except Exception as e:
                print(f"  ✗ Failed blown-up parallel ({obj_name}): {e}")
    
    def extract_all_individual_plots(self):
        """Extract all individual plots."""
        print(f"Extracting individual plots to subfolders...")
        print(f"- Contour maps: {self.contour_dir}")
        print(f"- Slice plots: {self.slice_dir}")
        print(f"- Blown-up parallel: {self.parallel_dir}")
        
        self.extract_individual_contour_maps()
        self.extract_individual_slice_plots()
        self.create_blown_up_parallel_plots()
        
        print(f"\n✓ Individual plot extraction complete!")
        
        # Count files in each directory
        contour_count = len([f for f in os.listdir(self.contour_dir) if f.endswith('.png')])
        slice_count = len([f for f in os.listdir(self.slice_dir) if f.endswith('.png')])
        parallel_count = len([f for f in os.listdir(self.parallel_dir) if f.endswith('.png')])
        
        print(f"- {contour_count} individual contour maps")
        print(f"- {slice_count} individual slice plots")
        print(f"- {parallel_count} blown-up parallel coordinate plots")


def main():
    """Main function."""
    print("Individual Optuna Plot Extractor")
    print("=" * 40)
    
    # Path to the optimization results
    results_file = "../../Agent_Storage/Hyperparameters/optimization_20250529_010111/optimization_results.json"
    
    if not os.path.exists(results_file):
        print(f"Results file not found: {results_file}")
        return
    
    # Create extractor
    extractor = IndividualOptunaPlotExtractor(results_file, output_dir="optuna_initial")
    
    # Extract all individual plots
    extractor.extract_all_individual_plots()


if __name__ == "__main__":
    main() 