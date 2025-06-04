#!/usr/bin/env python3
"""Extract individual Optuna plots organized in subfolders."""

import os
import json
import yaml
import optuna
import plotly.io as pio
from itertools import combinations

# Direct path that works
RESULTS_FILE = "../Agent_Storage/Hyperparameters/optimization_20250529_010111/optimization_results.json"
CONFIG_FILE = "../Agent_Storage/Hyperparameters/optimization_20250529_010111/optuna_config.yaml"
OUTPUT_DIR = "optuna_initial"

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
    contour_dir = os.path.join(OUTPUT_DIR, "individual_contours")
    slice_dir = os.path.join(OUTPUT_DIR, "individual_slices") 
    parallel_dir = os.path.join(OUTPUT_DIR, "blown_up_parallel")
    
    for folder in [contour_dir, slice_dir, parallel_dir]:
        os.makedirs(folder, exist_ok=True)
    
    return contour_dir, slice_dir, parallel_dir

def extract_contour_maps(study, contour_dir):
    """Extract individual contour maps with tiny points."""
    print("Extracting individual contour maps...")
    
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
                
                # Make points tiny and transparent
                fig.update_traces(
                    marker=dict(size=1, opacity=0.2),
                    selector=dict(type='scatter')
                )
                
                clean_param1 = param1.replace('model.', '')
                clean_param2 = param2.replace('model.', '')
                
                fig.update_layout(
                    title=f"Contour: {clean_param1} vs {clean_param2} ({obj_title})",
                    width=1000, height=800, showlegend=False
                )
                
                filename = f"contour_{obj_name}_{clean_param1}_vs_{clean_param2}.png"
                pio.write_image(fig, os.path.join(contour_dir, filename))
                count += 1
                print(f"  ✓ {count}: {filename}")
                
            except Exception as e:
                print(f"  ✗ Failed: {e}")
    
    print(f"Generated {count} contour maps")

def extract_slice_plots(study, slice_dir):
    """Extract individual slice plots."""
    print("Extracting individual slice plots...")
    
    import optuna.visualization as viz
    param_names = list(study.trials[0].params.keys())
    objectives = [
        ("goal", lambda t: t.values[0], "Goal Reached"),
        ("lava", lambda t: t.values[1], "Lava Proportion")
    ]
    
    count = 0
    for obj_name, obj_func, obj_title in objectives:
        for param_name in param_names:
            try:
                fig = viz.plot_slice(study, params=[param_name], target=obj_func)
                
                clean_param = param_name.replace('model.', '')
                fig.update_layout(
                    title=f"Slice: {clean_param} vs {obj_title}",
                    width=1000, height=700
                )
                
                filename = f"slice_{obj_name}_{clean_param}.png"
                pio.write_image(fig, os.path.join(slice_dir, filename))
                count += 1
                print(f"  ✓ {count}: {filename}")
                
            except Exception as e:
                print(f"  ✗ Failed: {e}")
    
    print(f"Generated {count} slice plots")

def create_blown_up_parallel(study, parallel_dir):
    """Create blown-up parallel coordinate plots."""
    print("Creating blown-up parallel coordinate plots...")
    
    import optuna.visualization as viz
    objectives = [
        ("goal", lambda t: t.values[0], "Goal Reached"),
        ("lava", lambda t: t.values[1], "Lava Proportion")
    ]
    
    count = 0
    for obj_name, obj_func, obj_title in objectives:
        try:
            fig = viz.plot_parallel_coordinate(study, target=obj_func)
            
            fig.update_layout(
                title=f"Blown-Up Parallel: {obj_title}",
                width=1600, height=1000,
                font=dict(size=16), title_font=dict(size=20)
            )
            
            filename = f"parallel_blown_up_{obj_name}.png"
            pio.write_image(fig, os.path.join(parallel_dir, filename))
            count += 1
            print(f"  ✓ {count}: {filename}")
            
        except Exception as e:
            print(f"  ✗ Failed: {e}")
    
    print(f"Generated {count} blown-up parallel plots")

def main():
    print("EXTRACTING INDIVIDUAL OPTUNA PLOTS")
    print("=" * 50)
    
    if not os.path.exists(RESULTS_FILE):
        print(f"ERROR: Results file not found: {RESULTS_FILE}")
        return
    
    print("✓ Loading data...")
    results, config = load_data()
    
    print(f"✓ Creating study from {len(results['all_trials'])} trials...")
    study = create_study(results, config)
    
    print("✓ Creating folders...")
    contour_dir, slice_dir, parallel_dir = create_folders()
    
    # Extract all plots
    extract_contour_maps(study, contour_dir)
    extract_slice_plots(study, slice_dir)
    create_blown_up_parallel(study, parallel_dir)
    
    print("\n" + "="*50)
    print("✓ EXTRACTION COMPLETE!")
    
    # Count results
    contour_count = len([f for f in os.listdir(contour_dir) if f.endswith('.png')])
    slice_count = len([f for f in os.listdir(slice_dir) if f.endswith('.png')])
    parallel_count = len([f for f in os.listdir(parallel_dir) if f.endswith('.png')])
    
    print(f"  - {contour_count} contour maps in {contour_dir}")
    print(f"  - {slice_count} slice plots in {slice_dir}")
    print(f"  - {parallel_count} blown-up parallel plots in {parallel_dir}")

if __name__ == "__main__":
    main() 