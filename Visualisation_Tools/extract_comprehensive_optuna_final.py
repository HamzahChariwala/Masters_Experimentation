#!/usr/bin/env python3
"""Extract comprehensive Optuna plots excluding BOTH problematic 20250526 optimization runs."""

import os
import json
import yaml
import optuna
import plotly.io as pio
import plotly.graph_objects as go
import glob
from itertools import combinations

# Paths to only the two clean optimization runs (excluding both 20250526 runs)
RESULTS_20250529 = "../Agent_Storage/Hyperparameters/optimization_20250529_010111/optimization_results.json"
RESULTS_20250528_DIR = "../Agent_Storage/Hyperparameters/optimization_20250528_020151/"

CONFIG_FILE = "../Agent_Storage/Hyperparameters/optimization_20250529_010111/optuna_config.yaml"
OUTPUT_DIR = "optuna_comprehensive"

def load_consolidated_results(json_path, trial_offset=0, run_name=""):
    """Load consolidated results from a JSON file."""
    if not os.path.exists(json_path):
        print(f"  Warning: {json_path} not found")
        return []
        
    with open(json_path, 'r') as f:
        results = json.load(f)
    
    trials = results.get('all_trials', [])
    
    # Add offset to trial numbers to distinguish runs
    for trial in trials:
        if 'number' in trial:
            trial['number'] += trial_offset
            
    print(f"  Loaded {len(trials)} trials from {run_name}")
    return trials

def load_individual_trials(trial_dir, trial_offset=0, run_name=""):
    """Load individual trial results from directory structure."""
    trials = []
    trial_dirs = glob.glob(os.path.join(trial_dir, "trial_*"))
    
    print(f"  Found {len(trial_dirs)} trial directories in {run_name}")
    
    for trial_path in sorted(trial_dirs):
        try:
            # Extract trial number from directory name
            trial_num = int(trial_path.split('_')[-2])
            
            # Load trial parameters
            params_file = os.path.join(trial_path, "trial_params.json")
            if not os.path.exists(params_file):
                continue
                
            with open(params_file, 'r') as f:
                params = json.load(f)
            
            # Load performance results
            perf_file = os.path.join(trial_path, "agent_3/evaluation_summary/performance_all_states.json")
            if not os.path.exists(perf_file):
                continue
                
            with open(perf_file, 'r') as f:
                performance = json.load(f)
            
            # Extract objectives
            goal_reached = performance.get('goal_reached_proportion', 0.0)
            lava_proportion = performance.get('next_cell_lava_proportion', 0.0)
            
            trial_data = {
                'number': trial_num + trial_offset,
                'state': 'COMPLETE',
                'values': [goal_reached, lava_proportion],
                'params': params
            }
            
            trials.append(trial_data)
            
        except Exception as e:
            print(f"    Warning: Could not load trial from {trial_path}: {e}")
            continue
    
    print(f"  Successfully loaded {len(trials)} trials from {run_name}")
    return trials

def load_all_optimization_data():
    """Load data from only the two clean optimization runs."""
    print("✓ Loading data from ONLY clean optimization runs (excluding BOTH 20250526 runs)...")
    
    all_trials = []
    
    # Load 20250529 (consolidated JSON)
    trials_20250529 = load_consolidated_results(
        RESULTS_20250529, trial_offset=0, run_name="20250529 (consolidated)"
    )
    all_trials.extend(trials_20250529)
    
    # Load 20250528 (individual trials)
    trials_20250528 = load_individual_trials(
        RESULTS_20250528_DIR, trial_offset=1000, run_name="20250528 (individual)"
    )
    all_trials.extend(trials_20250528)
    
    # EXCLUDED both 20250526 runs due to incompatible parameter bounds
    
    print(f"  Total combined trials: {len(all_trials)}")
    print("  Excluded: Both 20250526_013800 and 20250526_223042 (incompatible bounds)")
    return all_trials

def load_config():
    """Load configuration."""
    config = {}
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, 'r') as f:
            config = yaml.safe_load(f)
    return config

def create_comprehensive_study(all_trials, config):
    """Create Optuna study from all combined trials."""
    study = optuna.create_study(directions=['maximize', 'minimize'])
    
    # Create distributions - use flexible ranges to accommodate different bounds
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
    
    # If no config available, infer distributions from data
    if not distributions and all_trials:
        print("  Inferring parameter distributions from data...")
        param_ranges = {}
        for trial in all_trials:
            if trial['state'] == 'COMPLETE' and 'params' in trial:
                for param_name, value in trial['params'].items():
                    if param_name not in param_ranges:
                        param_ranges[param_name] = {'min': value, 'max': value, 'values': []}
                    param_ranges[param_name]['min'] = min(param_ranges[param_name]['min'], value)
                    param_ranges[param_name]['max'] = max(param_ranges[param_name]['max'], value)
                    param_ranges[param_name]['values'].append(value)
        
        for param_name, ranges in param_ranges.items():
            # Determine if integer or float based on values
            if all(isinstance(v, int) for v in ranges['values'][:10]):
                distributions[param_name] = optuna.distributions.IntDistribution(
                    low=ranges['min'], high=ranges['max']
                )
            else:
                distributions[param_name] = optuna.distributions.FloatDistribution(
                    low=ranges['min'], high=ranges['max']
                )
    
    # Add trials
    successful_trials = 0
    for trial_data in all_trials:
        if trial_data['state'] == 'COMPLETE' and trial_data['values'] is not None:
            values = trial_data['values']
            if isinstance(values, list) and len(values) == 2:
                try:
                    trial = optuna.trial.create_trial(
                        params=trial_data['params'],
                        distributions=distributions,
                        values=values
                    )
                    study.add_trial(trial)
                    successful_trials += 1
                except Exception as e:
                    print(f"    Warning: Could not add trial {trial_data.get('number', '?')}: {e}")
    
    print(f"  Successfully added {successful_trials} trials to study")
    return study

def create_folders():
    """Create output folders."""
    contour_dir = os.path.join(OUTPUT_DIR, "comprehensive_contours")
    slice_dir = os.path.join(OUTPUT_DIR, "comprehensive_slices")
    parallel_dir = os.path.join(OUTPUT_DIR, "comprehensive_parallel")
    importance_dir = os.path.join(OUTPUT_DIR, "comprehensive_importance")
    history_dir = os.path.join(OUTPUT_DIR, "comprehensive_history")
    
    for folder in [contour_dir, slice_dir, parallel_dir, importance_dir, history_dir]:
        os.makedirs(folder, exist_ok=True)
    
    return contour_dir, slice_dir, parallel_dir, importance_dir, history_dir

def extract_magma_contour_maps(study, contour_dir):
    """Extract comprehensive contour maps using magma colormap."""
    print("Extracting comprehensive magma contour maps...")
    
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
                    title=f"Clean Magma Heat Map: {clean_param1} vs {clean_param2} ({obj_title})",
                    width=1000, height=800, 
                    showlegend=False,
                    coloraxis=dict(colorscale='magma')
                )
                
                filename = f"clean_magma_{obj_name}_{clean_param1}_vs_{clean_param2}.png"
                pio.write_image(fig, os.path.join(contour_dir, filename))
                count += 1
                print(f"  ✓ {count}: {filename}")
                
            except Exception as e:
                print(f"  ✗ Failed: {e}")
    
    print(f"Generated {count} clean contour maps")

def extract_slice_plots(study, slice_dir):
    """Extract comprehensive slice plots."""
    print("Extracting clean slice plots...")
    
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
                    title=f"Clean Slice: {clean_param} vs {obj_title}",
                    width=1000, height=700
                )
                
                filename = f"clean_slice_{obj_name}_{clean_param}.png"
                pio.write_image(fig, os.path.join(slice_dir, filename))
                count += 1
                print(f"  ✓ {count}: {filename}")
                
            except Exception as e:
                print(f"  ✗ Failed: {e}")
    
    print(f"Generated {count} clean slice plots")

def create_parallel_plots(study, parallel_dir):
    """Create comprehensive parallel coordinate plots with magma."""
    print("Creating clean parallel coordinate plots with magma...")
    
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
                title=f"Clean Magma Parallel: {obj_title}",
                width=1600, height=1000,
                font=dict(size=16), title_font=dict(size=20),
                coloraxis=dict(colorscale='magma')
            )
            
            filename = f"clean_parallel_{obj_name}.png"
            pio.write_image(fig, os.path.join(parallel_dir, filename))
            count += 1
            print(f"  ✓ {count}: {filename}")
            
        except Exception as e:
            print(f"  ✗ Failed: {e}")
    
    print(f"Generated {count} clean parallel plots")

def create_parameter_importance_plots(study, importance_dir):
    """Create clean parameter importance plots with consistent magma colors."""
    print("Creating clean parameter importance plots with window sizing colors...")
    
    import optuna.visualization as viz
    import matplotlib.pyplot as plt
    
    objectives = [
        ("goal", lambda t: t.values[0], "Goal Reached", 0.3),      # Purple shade like window sizing
        ("lava", lambda t: t.values[1], "Lava Proportion", 0.6)   # Magenta shade like window sizing
    ]
    
    count = 0
    for obj_name, obj_func, obj_title, magma_value in objectives:
        try:
            fig = viz.plot_param_importances(study, target=obj_func)
            
            # Get the magma color for this objective
            magma_color = plt.cm.magma(magma_value)
            magma_hex = f"#{int(magma_color[0]*255):02x}{int(magma_color[1]*255):02x}{int(magma_color[2]*255):02x}"
            
            # Apply consistent magma color to all bars
            fig.update_traces(
                marker=dict(
                    color=magma_hex,
                    line=dict(color=magma_hex, width=1)
                ),
                selector=dict(type='bar')
            )
            
            # Clean parameter names
            if fig.data[0].y is not None:
                clean_labels = [label.replace('model.', '') for label in fig.data[0].y]
                fig.update_traces(y=clean_labels)
            
            fig.update_layout(
                title=f"Clean Parameter Importance: {obj_title}",
                width=1000, height=800,
                font=dict(size=14), title_font=dict(size=18),
                xaxis_title="Importance",
                yaxis_title="Parameters",
                plot_bgcolor='white',
                paper_bgcolor='white'
            )
            
            filename = f"clean_importance_{obj_name}.png"
            pio.write_image(fig, os.path.join(importance_dir, filename))
            count += 1
            print(f"  ✓ {count}: {filename} (magma {magma_value})")
            
        except Exception as e:
            print(f"  ✗ Failed parameter importance for {obj_name}: {e}")
    
    print(f"Generated {count} clean parameter importance plots")

def create_optimization_history_plots(study, history_dir):
    """Create clean optimization history plots with magma."""
    print("Creating clean optimization history plots with magma...")
    
    import optuna.visualization as viz
    objectives = [
        ("goal", lambda t: t.values[0], "Goal Reached"),
        ("lava", lambda t: t.values[1], "Lava Proportion")
    ]
    
    count = 0
    for obj_name, obj_func, obj_title in objectives:
        try:
            fig = viz.plot_optimization_history(study, target=obj_func)
            
            # Apply magma colorscale to traces
            for i, trace in enumerate(fig.data):
                if hasattr(trace, 'line'):
                    # Use different magma values for different traces
                    magma_color = f"rgba({int(255*0.2)}, {int(255*0.1)}, {int(255*0.5)}, 0.8)"
                    trace.line.color = magma_color
                    trace.line.width = 3
                elif hasattr(trace, 'marker'):
                    # For scatter points
                    trace.marker.color = f"rgba({int(255*0.8)}, {int(255*0.2)}, {int(255*0.9)}, 0.7)"
                    trace.marker.size = 8
            
            fig.update_layout(
                title=f"Clean Optimization History: {obj_title}",
                width=1200, height=700,
                font=dict(size=14), title_font=dict(size=18),
                xaxis_title="Trial Number",
                yaxis_title=obj_title,
                plot_bgcolor='white',
                paper_bgcolor='white'
            )
            
            filename = f"clean_history_{obj_name}.png"
            pio.write_image(fig, os.path.join(history_dir, filename))
            count += 1
            print(f"  ✓ {count}: {filename}")
            
        except Exception as e:
            print(f"  ✗ Failed optimization history for {obj_name}: {e}")
    
    print(f"Generated {count} clean optimization history plots")

def main():
    print("EXTRACTING TRULY CLEAN OPTUNA PLOTS (2 OPTIMIZATION RUNS ONLY)")
    print("Excluding BOTH 20250526 runs - using only consistent parameter bounds")
    print("=" * 80)
    
    # Load all optimization data
    all_trials = load_all_optimization_data()
    
    if not all_trials:
        print("ERROR: No trials loaded from any optimization run!")
        return
    
    print("✓ Loading configuration...")
    config = load_config()
    
    print("✓ Creating clean study...")
    study = create_comprehensive_study(all_trials, config)
    total_trials = len(study.trials)
    print(f"  Clean study has {total_trials} total trials")
    
    if total_trials == 0:
        print("ERROR: No valid trials in study!")
        return
    
    print("✓ Creating output folders...")
    contour_dir, slice_dir, parallel_dir, importance_dir, history_dir = create_folders()
    
    # Extract all plots
    extract_magma_contour_maps(study, contour_dir)
    extract_slice_plots(study, slice_dir)
    create_parallel_plots(study, parallel_dir)
    create_parameter_importance_plots(study, importance_dir)
    create_optimization_history_plots(study, history_dir)
    
    print("\n" + "="*80)
    print("✓ TRULY CLEAN EXTRACTION COMPLETE!")
    
    # Count results
    contour_count = len([f for f in os.listdir(contour_dir) if f.endswith('.png')])
    slice_count = len([f for f in os.listdir(slice_dir) if f.endswith('.png')])
    parallel_count = len([f for f in os.listdir(parallel_dir) if f.endswith('.png')])
    importance_count = len([f for f in os.listdir(importance_dir) if f.endswith('.png')])
    history_count = len([f for f in os.listdir(history_dir) if f.endswith('.png')])
    
    print(f"  - {contour_count} clean contour maps in {contour_dir}")
    print(f"  - {slice_count} clean slice plots in {slice_dir}")
    print(f"  - {parallel_count} clean parallel plots in {parallel_dir}")
    print(f"  - {importance_count} clean parameter importance plots in {importance_dir}")
    print(f"  - {history_count} clean optimization history plots in {history_dir}")
    print(f"  - Total trials analyzed: {total_trials}")
    print(f"  - Data sources: 2 optimization runs ONLY (20250529 + 20250528)")
    print(f"  - Excluded: BOTH 20250526 runs (incompatible parameter bounds)")
    print(f"  - Importance colors: Goal magma(0.3), Lava magma(0.6)")

if __name__ == "__main__":
    main() 