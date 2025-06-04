#!/usr/bin/env python3
"""Extract Optuna plots with magma colormap - no contour lines, just color maps."""

import os
import json
import yaml
import optuna
import plotly.io as pio
import plotly.graph_objects as go
from itertools import combinations
import numpy as np

# Direct path that works
RESULTS_FILE = "../Agent_Storage/Hyperparameters/optimization_20250529_010111/optimization_results.json"
CONFIG_FILE = "../Agent_Storage/Hyperparameters/optimization_20250529_010111/optuna_config.yaml"
OUTPUT_DIR = "optuna_updated"

def load_data():
    """Load results and config."""
    with open(RESULTS_FILE, 'r') as f:
        results = json.load(f)
    
    config = {}
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, 'r') as f:
            config = yaml.safe_load(f)
    
    return results, config

def create_study(results, config):
    """Create Optuna study from results."""
    study = optuna.create_study(directions=['maximize', 'minimize'])
    
    # Create distributions
    distributions = {}
    if 'bayesian_optimization' in config:
        for param_name, param_config in config['bayesian_optimization'].items():
            if param_config['distribution'] == 'int_uniform':
                distributions[param_name] = optuna.distributions.IntDistribution(
                    low=param_config['min'], high=param_config['max']
                )
            elif param_config['distribution'] == 'uniform':
                distributions[param_name] = optuna.distributions.FloatDistribution(
                    low=param_config['min'], high=param_config['max']
                )
            elif param_config['distribution'] == 'loguniform':
                distributions[param_name] = optuna.distributions.FloatDistribution(
                    low=param_config['min'], high=param_config['max'], log=True
                )
    
    # Add trials
    for trial_data in results['all_trials']:
        if trial_data['state'] == 'COMPLETE' and trial_data['values'] is not None:
            values = trial_data['values']
            if isinstance(values, list) and len(values) == 2:
                trial = optuna.trial.create_trial(
                    params=trial_data['params'],
                    distributions=distributions,
                    values=values
                )
                study.add_trial(trial)
    
    return study

def create_folders():
    """Create output folders."""
    contour_dir = os.path.join(OUTPUT_DIR, "magma_contours")
    parallel_dir = os.path.join(OUTPUT_DIR, "magma_parallel")
    
    for folder in [contour_dir, parallel_dir]:
        os.makedirs(folder, exist_ok=True)
    
    return contour_dir, parallel_dir

def extract_magma_contour_maps(study, contour_dir):
    """Extract contour maps without contour lines using magma colormap."""
    print("Extracting magma contour maps (no contour lines)...")
    
    import optuna.visualization as viz
    param_names = list(study.trials[0].params.keys())
    objectives = [
        ("goal", lambda t: t.values[0], "Goal Reached"),
        ("lava", lambda t: t.values[1], "Lava Proportion")
    ]
    
    count = 0
    for obj_name, obj_func, obj_title in objectives:
        for param1, param2 in combinations(param_names, 2):
            try:
                fig = viz.plot_contour(study, params=[param1, param2], target=obj_func)
                
                # Remove contour lines and use magma colormap
                fig.update_traces(
                    # Remove scatter points completely
                    marker=dict(size=0, opacity=0),
                    selector=dict(type='scatter')
                )
                
                # Update contour traces to use magma and remove lines
                fig.update_traces(
                    colorscale='magma',
                    contours=dict(
                        showlines=False,  # Remove contour lines
                        coloring='fill'   # Only show filled areas
                    ),
                    selector=dict(type='contour')
                )
                
                # Update heatmap traces to use magma
                fig.update_traces(
                    colorscale='magma',
                    selector=dict(type='heatmap')
                )
                
                clean_param1 = param1.replace('model.', '')
                clean_param2 = param2.replace('model.', '')
                
                fig.update_layout(
                    title=f"Magma Heat Map: {clean_param1} vs {clean_param2} ({obj_title})",
                    width=1000, height=800, 
                    showlegend=False,
                    coloraxis=dict(colorscale='magma')
                )
                
                filename = f"magma_contour_{obj_name}_{clean_param1}_vs_{clean_param2}.png"
                pio.write_image(fig, os.path.join(contour_dir, filename))
                count += 1
                print(f"  ✓ {count}: {filename}")
                
            except Exception as e:
                print(f"  ✗ Failed: {e}")
    
    print(f"Generated {count} magma contour maps")

def create_magma_parallel_plots(study, parallel_dir):
    """Create blown-up parallel coordinate plots with magma colormap."""
    print("Creating blown-up parallel coordinate plots with magma...")
    
    import optuna.visualization as viz
    objectives = [
        ("goal", lambda t: t.values[0], "Goal Reached"),
        ("lava", lambda t: t.values[1], "Lava Proportion")
    ]
    
    count = 0
    for obj_name, obj_func, obj_title in objectives:
        try:
            fig = viz.plot_parallel_coordinate(study, target=obj_func)
            
            # Update to use magma colorscale
            fig.update_traces(
                line=dict(colorscale='magma'),
                selector=dict(type='parcoords')
            )
            
            fig.update_layout(
                title=f"Magma Parallel Coordinate Plot: {obj_title}",
                width=1600, height=1000,
                font=dict(size=16), title_font=dict(size=20),
                coloraxis=dict(colorscale='magma')
            )
            
            filename = f"magma_parallel_{obj_name}.png"
            pio.write_image(fig, os.path.join(parallel_dir, filename))
            count += 1
            print(f"  ✓ {count}: {filename}")
            
        except Exception as e:
            print(f"  ✗ Failed: {e}")
    
    print(f"Generated {count} magma parallel plots")

def main():
    print("EXTRACTING MAGMA OPTUNA PLOTS")
    print("=" * 50)
    
    if not os.path.exists(RESULTS_FILE):
        print(f"ERROR: Results file not found: {RESULTS_FILE}")
        return
    
    print("✓ Loading data...")
    results, config = load_data()
    
    print(f"✓ Creating study from {len(results['all_trials'])} trials...")
    study = create_study(results, config)
    
    print("✓ Creating optuna_updated folders...")
    contour_dir, parallel_dir = create_folders()
    
    # Extract magma plots
    extract_magma_contour_maps(study, contour_dir)
    create_magma_parallel_plots(study, parallel_dir)
    
    print("\n" + "="*50)
    print("✓ MAGMA EXTRACTION COMPLETE!")
    
    # Count results
    contour_count = len([f for f in os.listdir(contour_dir) if f.endswith('.png')])
    parallel_count = len([f for f in os.listdir(parallel_dir) if f.endswith('.png')])
    
    print(f"  - {contour_count} magma contour maps (no lines) in {contour_dir}")
    print(f"  - {parallel_count} magma parallel plots in {parallel_dir}")

if __name__ == "__main__":
    main() 