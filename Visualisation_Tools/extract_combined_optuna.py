#!/usr/bin/env python3
"""Extract Optuna plots combining trials from multiple optimization runs."""

import os
import json
import yaml
import optuna
import plotly.io as pio
import glob
from itertools import combinations

# Paths to both optimization runs
RESULTS_20250529 = "../Agent_Storage/Hyperparameters/optimization_20250529_010111/optimization_results.json"
RESULTS_20250528_DIR = "../Agent_Storage/Hyperparameters/optimization_20250528_020151/"
CONFIG_FILE = "../Agent_Storage/Hyperparameters/optimization_20250529_010111/optuna_config.yaml"
OUTPUT_DIR = "optuna_combined"

def load_20250529_data():
    """Load consolidated results from 20250529."""
    with open(RESULTS_20250529, 'r') as f:
        results = json.load(f)
    return results['all_trials']

def load_20250528_data():
    """Load individual trial results from 20250528."""
    trials = []
    trial_dirs = glob.glob(os.path.join(RESULTS_20250528_DIR, "trial_*"))
    
    print(f"Found {len(trial_dirs)} trial directories in 20250528 run")
    
    for trial_dir in sorted(trial_dirs):
        try:
            # Extract trial number from directory name
            trial_num = int(trial_dir.split('_')[-2])
            
            # Load trial parameters
            params_file = os.path.join(trial_dir, "trial_params.json")
            if not os.path.exists(params_file):
                continue
                
            with open(params_file, 'r') as f:
                params = json.load(f)
            
            # Load performance results
            perf_file = os.path.join(trial_dir, "agent_3/evaluation_summary/performance_all_states.json")
            if not os.path.exists(perf_file):
                continue
                
            with open(perf_file, 'r') as f:
                performance = json.load(f)
            
            # Extract objectives
            goal_reached = performance.get('goal_reached_proportion', 0.0)
            lava_proportion = performance.get('next_cell_lava_proportion', 0.0)
            
            trial_data = {
                'number': trial_num + 1000,  # Offset to distinguish from 20250529 trials
                'state': 'COMPLETE',
                'values': [goal_reached, lava_proportion],
                'params': params
            }
            
            trials.append(trial_data)
            
        except Exception as e:
            print(f"  Warning: Could not load trial from {trial_dir}: {e}")
            continue
    
    print(f"Successfully loaded {len(trials)} trials from 20250528 run")
    return trials

def load_config():
    """Load configuration."""
    config = {}
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, 'r') as f:
            config = yaml.safe_load(f)
    return config

def create_combined_study(trials_20250529, trials_20250528, config):
    """Create Optuna study from combined trials."""
    all_trials = trials_20250529 + trials_20250528
    
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
    for trial_data in all_trials:
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
    contour_dir = os.path.join(OUTPUT_DIR, "combined_contours")
    slice_dir = os.path.join(OUTPUT_DIR, "combined_slices")
    parallel_dir = os.path.join(OUTPUT_DIR, "combined_parallel")
    
    for folder in [contour_dir, slice_dir, parallel_dir]:
        os.makedirs(folder, exist_ok=True)
    
    return contour_dir, slice_dir, parallel_dir

def extract_magma_contour_maps(study, contour_dir):
    """Extract contour maps without contour lines using magma colormap."""
    print("Extracting combined magma contour maps...")
    
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
                
                # Remove scatter points and contour lines, use magma
                fig.update_traces(
                    marker=dict(size=0, opacity=0),
                    selector=dict(type='scatter')
                )
                
                fig.update_traces(
                    colorscale='magma',
                    contours=dict(showlines=False, coloring='fill'),
                    selector=dict(type='contour')
                )
                
                fig.update_traces(
                    colorscale='magma',
                    selector=dict(type='heatmap')
                )
                
                clean_param1 = param1.replace('model.', '')
                clean_param2 = param2.replace('model.', '')
                
                fig.update_layout(
                    title=f"Combined Magma Heat Map: {clean_param1} vs {clean_param2} ({obj_title})",
                    width=1000, height=800, 
                    showlegend=False,
                    coloraxis=dict(colorscale='magma')
                )
                
                filename = f"combined_magma_{obj_name}_{clean_param1}_vs_{clean_param2}.png"
                pio.write_image(fig, os.path.join(contour_dir, filename))
                count += 1
                print(f"  ✓ {count}: {filename}")
                
            except Exception as e:
                print(f"  ✗ Failed: {e}")
    
    print(f"Generated {count} combined contour maps")

def extract_slice_plots(study, slice_dir):
    """Extract individual slice plots."""
    print("Extracting combined slice plots...")
    
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
                    title=f"Combined Slice: {clean_param} vs {obj_title}",
                    width=1000, height=700
                )
                
                filename = f"combined_slice_{obj_name}_{clean_param}.png"
                pio.write_image(fig, os.path.join(slice_dir, filename))
                count += 1
                print(f"  ✓ {count}: {filename}")
                
            except Exception as e:
                print(f"  ✗ Failed: {e}")
    
    print(f"Generated {count} combined slice plots")

def create_parallel_plots(study, parallel_dir):
    """Create blown-up parallel coordinate plots with magma."""
    print("Creating combined parallel coordinate plots with magma...")
    
    import optuna.visualization as viz
    objectives = [
        ("goal", lambda t: t.values[0], "Goal Reached"),
        ("lava", lambda t: t.values[1], "Lava Proportion")
    ]
    
    count = 0
    for obj_name, obj_func, obj_title in objectives:
        try:
            fig = viz.plot_parallel_coordinate(study, target=obj_func)
            
            fig.update_traces(
                line=dict(colorscale='magma'),
                selector=dict(type='parcoords')
            )
            
            fig.update_layout(
                title=f"Combined Magma Parallel: {obj_title}",
                width=1600, height=1000,
                font=dict(size=16), title_font=dict(size=20),
                coloraxis=dict(colorscale='magma')
            )
            
            filename = f"combined_parallel_{obj_name}.png"
            pio.write_image(fig, os.path.join(parallel_dir, filename))
            count += 1
            print(f"  ✓ {count}: {filename}")
            
        except Exception as e:
            print(f"  ✗ Failed: {e}")
    
    print(f"Generated {count} combined parallel plots")

def main():
    print("EXTRACTING COMBINED OPTUNA PLOTS (20250528 + 20250529)")
    print("=" * 60)
    
    if not os.path.exists(RESULTS_20250529):
        print(f"ERROR: 20250529 results not found: {RESULTS_20250529}")
        return
        
    if not os.path.exists(RESULTS_20250528_DIR):
        print(f"ERROR: 20250528 directory not found: {RESULTS_20250528_DIR}")
        return
    
    print("✓ Loading 20250529 consolidated data...")
    trials_20250529 = load_20250529_data()
    print(f"  Loaded {len(trials_20250529)} trials from 20250529")
    
    print("✓ Loading 20250528 individual trial data...")
    trials_20250528 = load_20250528_data()
    
    print("✓ Loading configuration...")
    config = load_config()
    
    print("✓ Creating combined study...")
    study = create_combined_study(trials_20250529, trials_20250528, config)
    total_trials = len(study.trials)
    print(f"  Combined study has {total_trials} total trials")
    
    print("✓ Creating output folders...")
    contour_dir, slice_dir, parallel_dir = create_folders()
    
    # Extract all plots
    extract_magma_contour_maps(study, contour_dir)
    extract_slice_plots(study, slice_dir)
    create_parallel_plots(study, parallel_dir)
    
    print("\n" + "="*60)
    print("✓ COMBINED EXTRACTION COMPLETE!")
    
    # Count results
    contour_count = len([f for f in os.listdir(contour_dir) if f.endswith('.png')])
    slice_count = len([f for f in os.listdir(slice_dir) if f.endswith('.png')])
    parallel_count = len([f for f in os.listdir(parallel_dir) if f.endswith('.png')])
    
    print(f"  - {contour_count} combined contour maps in {contour_dir}")
    print(f"  - {slice_count} combined slice plots in {slice_dir}")
    print(f"  - {parallel_count} combined parallel plots in {parallel_dir}")
    print(f"  - Total trials analyzed: {total_trials}")

if __name__ == "__main__":
    main() 