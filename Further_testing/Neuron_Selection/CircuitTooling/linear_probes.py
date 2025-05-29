#!/usr/bin/env python3
"""
Script: linear_probes.py

Train logistic regression models on clean and corrupt activations to create linear probes.
These probes can help identify which neurons are most predictive of clean vs corrupted behavior.

Also train linear regression models to predict logit values and margins.
"""
import argparse
import numpy as np
import os
import sys
import json
import pickle
from pathlib import Path
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Add the parent directory to the Python path to find other modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_activations(activations_file):
    """
    Load activations from a JSON file.
    
    Args:
        activations_file: Path to the activations JSON file
        
    Returns:
        dict: Dictionary of activations organized by input ID and layer
    """
    if not os.path.exists(activations_file):
        raise FileNotFoundError(f"Activations file not found: {activations_file}")
    
    with open(activations_file, 'r') as f:
        activations = json.load(f)
    
    return activations


def extract_layer_data(clean_activations, corrupt_activations, layer_name):
    """
    Extract activation data for a specific layer from both clean and corrupt activations.
    
    Args:
        clean_activations: Dictionary of clean activations
        corrupt_activations: Dictionary of corrupt activations
        layer_name: Name of the layer to extract (e.g., 'q_net.0')
        
    Returns:
        tuple: (X, y) where X is the activation data and y are the labels (0=clean, 1=corrupt)
    """
    X = []
    y = []
    
    # Get common input IDs between clean and corrupt
    clean_ids = set(clean_activations.keys())
    corrupt_ids = set(corrupt_activations.keys())
    common_ids = clean_ids.intersection(corrupt_ids)
    
    if not common_ids:
        raise ValueError("No common input IDs found between clean and corrupt activations")
    
    print(f"Found {len(common_ids)} common input IDs")
    
    # Extract clean activations
    for input_id in common_ids:
        if layer_name in clean_activations[input_id]:
            activation = clean_activations[input_id][layer_name]
            # Handle nested list structure (activations are usually nested)
            if isinstance(activation, list) and len(activation) > 0:
                if isinstance(activation[0], list):
                    activation = activation[0]  # Flatten if nested
                X.append(activation)
                y.append(0)  # Clean = 0
    
    # Extract corrupt activations
    for input_id in common_ids:
        if layer_name in corrupt_activations[input_id]:
            activation = corrupt_activations[input_id][layer_name]
            # Handle nested list structure
            if isinstance(activation, list) and len(activation) > 0:
                if isinstance(activation[0], list):
                    activation = activation[0]  # Flatten if nested
                X.append(activation)
                y.append(1)  # Corrupt = 1
    
    X = np.array(X)
    y = np.array(y)
    
    if len(X) == 0:
        raise ValueError(f"No activation data found for layer {layer_name}")
    
    print(f"Layer {layer_name}: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Clean samples: {np.sum(y == 0)}, Corrupt samples: {np.sum(y == 1)}")
    
    return X, y


def extract_logit_data(activations, layer_name):
    """
    Extract logit values from activations for linear regression.
    
    Args:
        activations: Dictionary of activations
        layer_name: Name of the layer to extract (e.g., 'q_net.0')
        
    Returns:
        tuple: (X, logits) where X is the activation data and logits are the q_net.4 values
    """
    X = []
    logits = []
    
    # Extract activations and corresponding logits
    for input_id, data in activations.items():
        if layer_name in data and 'q_net.4' in data:
            activation = data[layer_name]
            logit_values = data['q_net.4']
            
            # Handle nested list structure
            if isinstance(activation, list) and len(activation) > 0:
                if isinstance(activation[0], list):
                    activation = activation[0]  # Flatten if nested
            
            if isinstance(logit_values, list) and len(logit_values) > 0:
                if isinstance(logit_values[0], list):
                    logit_values = logit_values[0]  # Flatten if nested
            
            X.append(activation)
            logits.append(logit_values)
    
    X = np.array(X)
    logits = np.array(logits)
    
    if len(X) == 0:
        raise ValueError(f"No activation data found for layer {layer_name}")
    
    print(f"Layer {layer_name}: {X.shape[0]} samples, {X.shape[1]} features, {logits.shape[1]} logit outputs")
    
    return X, logits


def compute_logit_targets(logits):
    """
    Compute target values for logit regression models.
    
    Args:
        logits: Array of logit values (n_samples, n_actions)
        
    Returns:
        tuple: (top_logits, second_logits, margins) where each is a 1D array
    """
    # Sort logits in descending order for each sample
    sorted_logits = np.sort(logits, axis=1)[:, ::-1]
    
    # Extract top and second logits
    top_logits = sorted_logits[:, 0]
    second_logits = sorted_logits[:, 1]
    
    # Compute margin (difference between top two)
    margins = top_logits - second_logits
    
    return top_logits, second_logits, margins


def train_linear_regression(X, y, target_name, layer_name, test_size=0.2, random_state=42):
    """
    Train a linear regression model on the activation data.
    
    Args:
        X: Activation features
        y: Target values (continuous)
        target_name: Name of the target for logging (e.g., 'top_logit')
        layer_name: Name of the layer for logging
        test_size: Proportion of data to use for testing
        random_state: Random seed for reproducibility
        
    Returns:
        tuple: (model, scaler, metrics) where metrics contains training results
    """
    print(f"\nTraining linear regression for {target_name} on layer {layer_name}")
    
    # Check if dataset is too small for meaningful train/test split
    n_samples = X.shape[0]
    if n_samples < 20:
        print(f"⚠️  WARNING: Very small dataset ({n_samples} samples)")
        
        if n_samples < 10:
            print("⚠️  Dataset too small for train/test split. Using all data for training.")
            print("   NOTE: Test metrics will be the same as training metrics (overfitting likely)")
            X_train, X_test = X, X
            y_train, y_test = y, y
            use_split = False
        else:
            print(f"   Proceeding with {test_size:.0%} test split, but results may not be reliable.")
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )
            use_split = True
    else:
        # Normal case with sufficient data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        use_split = True
    
    print(f"Training set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train linear regression with L2 regularization (Ridge regression)
    # Using closed-form solution via normal equations when possible
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)
    
    # Evaluate the model
    train_pred = model.predict(X_train_scaled)
    test_pred = model.predict(X_test_scaled)
    
    train_mse = mean_squared_error(y_train, train_pred)
    test_mse = mean_squared_error(y_test, test_pred)
    train_r2 = r2_score(y_train, train_pred)
    test_r2 = r2_score(y_test, test_pred)
    
    print(f"Training MSE: {train_mse:.6f}, R²: {train_r2:.4f}")
    if use_split:
        print(f"Test MSE: {test_mse:.6f}, R²: {test_r2:.4f}")
    else:
        print(f"Test MSE: {test_mse:.6f}, R²: {test_r2:.4f} (same as training - no split used)")
    
    # Get feature importance (absolute values of coefficients)
    feature_importance = np.abs(model.coef_)
    
    # Create ranked list of all neuron indices with their weights
    neuron_rankings = []
    for neuron_idx in range(len(feature_importance)):
        neuron_rankings.append({
            'neuron_index': neuron_idx,
            'weight': float(feature_importance[neuron_idx]),
            'raw_coefficient': float(model.coef_[neuron_idx])
        })
    
    # Sort by absolute weight (descending)
    neuron_rankings.sort(key=lambda x: x['weight'], reverse=True)
    
    # Compile metrics
    metrics = {
        'target_name': target_name,
        'layer_name': layer_name,
        'train_mse': float(train_mse),
        'test_mse': float(test_mse),
        'train_r2': float(train_r2),
        'test_r2': float(test_r2),
        'n_features': X.shape[1],
        'n_samples': X.shape[0],
        'n_train_samples': len(X_train),
        'n_test_samples': len(X_test),
        'used_train_test_split': use_split,
        'dataset_size_warning': n_samples < 20,
        'neuron_rankings': neuron_rankings,
        'model_type': 'linear_regression',
        'solver_info': {
            'solver': 'normal_equations',
            'converged': True,  # Linear regression always converges with normal equations
            'method': 'closed_form'
        }
    }
    
    return model, scaler, metrics


def save_regression_results(model, scaler, metrics, output_dir, layer_name, target_name):
    """
    Save trained linear regression model, scaler, and metrics to files.
    
    Args:
        model: Trained LinearRegression model
        scaler: Fitted StandardScaler
        metrics: Dictionary of training metrics
        output_dir: Directory to save files
        layer_name: Name of the layer (e.g., 'q_net.0')
        target_name: Name of the target (e.g., 'top_logit')
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create filename prefix
    prefix = f"{layer_name}_{target_name}"
    
    # Save model
    model_file = output_dir / f"{prefix}_model.pkl"
    with open(model_file, 'wb') as f:
        pickle.dump(model, f)
    
    # Save scaler
    scaler_file = output_dir / f"{prefix}_scaler.pkl"
    with open(scaler_file, 'wb') as f:
        pickle.dump(scaler, f)
    
    # Save metrics
    metrics_file = output_dir / f"{prefix}_metrics.json"
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"Saved {target_name} regression for {layer_name}:")
    print(f"  Model: {model_file}")
    print(f"  Scaler: {scaler_file}")
    print(f"  Metrics: {metrics_file}")


def train_probe(X, y, layer_name, test_size=0.2, random_state=42):
    """
    Train a logistic regression probe on the activation data.
    
    Args:
        X: Activation features
        y: Labels (0=clean, 1=corrupt)
        layer_name: Name of the layer for logging
        test_size: Proportion of data to use for testing
        random_state: Random seed for reproducibility
        
    Returns:
        tuple: (model, scaler, metrics) where metrics contains training results
    """
    print(f"\nTraining probe for layer {layer_name}")
    
    # Check if dataset is too small for meaningful train/test split
    n_samples = X.shape[0]
    if n_samples < 20:
        print(f"⚠️  WARNING: Very small dataset ({n_samples} samples)")
        print(f"   Clean samples: {np.sum(y == 0)}, Corrupt samples: {np.sum(y == 1)}")
        
        if n_samples < 10:
            print("⚠️  Dataset too small for train/test split. Using all data for training.")
            print("   NOTE: Test accuracy will be the same as training accuracy (overfitting likely)")
            X_train, X_test = X, X
            y_train, y_test = y, y
            use_split = False
        else:
            print(f"   Proceeding with {test_size:.0%} test split, but results may not be reliable.")
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state, stratify=y
            )
            use_split = True
    else:
        # Normal case with sufficient data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        use_split = True
    
    print(f"Training set: {len(X_train)} samples (Clean: {np.sum(y_train == 0)}, Corrupt: {np.sum(y_train == 1)})")
    print(f"Test set: {len(X_test)} samples (Clean: {np.sum(y_test == 0)}, Corrupt: {np.sum(y_test == 1)})")
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train logistic regression with L2 regularization
    model = LogisticRegression(
        random_state=random_state,
        max_iter=10000,  # Increased iterations for better convergence
        solver='lbfgs',  # L-BFGS: quasi-Newton method that guarantees convergence to global optimum
        penalty='l2',  # L2 regularization
        C=1.0,  # Regularization strength (lower values = stronger regularization)
        tol=1e-8  # Tighter tolerance for better convergence
    )
    
    model.fit(X_train_scaled, y_train)
    
    # Check convergence
    n_iter_used = getattr(model, 'n_iter_', [None])[0]
    converged = n_iter_used is not None and n_iter_used < 10000
    
    print(f"Solver convergence: {'✓ Converged' if converged else '✗ Did not converge'}")
    if n_iter_used is not None:
        print(f"Iterations used: {n_iter_used}/10000")
    
    # Evaluate the model
    train_pred = model.predict(X_train_scaled)
    test_pred = model.predict(X_test_scaled)
    
    train_acc = accuracy_score(y_train, train_pred)
    test_acc = accuracy_score(y_test, test_pred)
    
    print(f"Training accuracy: {train_acc:.4f}")
    if use_split:
        print(f"Test accuracy: {test_acc:.4f}")
    else:
        print(f"Test accuracy: {test_acc:.4f} (same as training - no split used)")
    
    # Get feature importance (absolute values of coefficients)
    feature_importance = np.abs(model.coef_[0])
    
    # Create ranked list of all neuron indices with their weights
    neuron_rankings = []
    for neuron_idx in range(len(feature_importance)):
        neuron_rankings.append({
            'neuron_index': neuron_idx,
            'weight': float(feature_importance[neuron_idx]),
            'raw_coefficient': float(model.coef_[0][neuron_idx])
        })
    
    # Sort by absolute weight (descending)
    neuron_rankings.sort(key=lambda x: x['weight'], reverse=True)
    
    # Get top 10 for backwards compatibility
    top_features = np.argsort(feature_importance)[::-1][:10]
    
    metrics = {
        'layer_name': layer_name,
        'train_accuracy': train_acc,
        'test_accuracy': test_acc,
        'n_features': X.shape[1],
        'n_samples': X.shape[0],
        'n_train_samples': len(X_train),
        'n_test_samples': len(X_test),
        'used_train_test_split': use_split,
        'dataset_size_warning': n_samples < 20,
        'regularization': {
            'penalty': 'l2',
            'C': 1.0
        },
        'solver_info': {
            'solver': 'lbfgs',
            'max_iter': 10000,
            'tolerance': 1e-8,
            'n_iter': int(getattr(model, 'n_iter_', [None])[0]) if getattr(model, 'n_iter_', [None])[0] is not None else None,  # Convert to Python int
            'converged': bool(getattr(model, 'n_iter_', [0])[0] < 10000)  # Convert to Python bool
        },
        'neuron_rankings': neuron_rankings,  # Full ranked list with weights
        'top_features': top_features.tolist(),  # Keep for backwards compatibility
        'feature_importance': feature_importance.tolist(),  # Keep for backwards compatibility
        'classification_report': classification_report(y_test, test_pred, output_dict=True)
    }
    
    print(f"Top 10 most important neurons: {[ranking['neuron_index'] for ranking in neuron_rankings[:10]]}")
    print(f"Top neuron (index {neuron_rankings[0]['neuron_index']}) weight: {neuron_rankings[0]['weight']:.6f}")
    
    return model, scaler, metrics


def save_probe_results(model, scaler, metrics, output_dir, layer_name):
    """
    Save the trained probe model, scaler, and metrics to files.
    
    Args:
        model: Trained LogisticRegression model
        scaler: Fitted StandardScaler
        metrics: Dictionary of training metrics
        output_dir: Directory to save files
        layer_name: Name of the layer (used in filenames)
    """
    # Create clean_vs_corrupt subfolder for logistic regression results
    probe_output_dir = Path(output_dir) / "clean_vs_corrupt"
    probe_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save model
    model_file = probe_output_dir / f"{layer_name}_probe_model.pkl"
    with open(model_file, 'wb') as f:
        pickle.dump(model, f)
    
    # Save scaler
    scaler_file = probe_output_dir / f"{layer_name}_probe_scaler.pkl"
    with open(scaler_file, 'wb') as f:
        pickle.dump(scaler, f)
    
    # Save metrics
    metrics_file = probe_output_dir / f"{layer_name}_probe_metrics.json"
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"Saved probe for {layer_name} to {probe_output_dir}")


def get_layer_names_from_activations(activations):
    """
    Extract layer names from activation data, excluding the logits layer.
    
    Args:
        activations: Dictionary of activations
        
    Returns:
        list: List of layer names to train probes on
    """
    # Get a sample entry to examine available layers
    sample_id = next(iter(activations.keys()))
    all_layers = list(activations[sample_id].keys())
    
    # Filter out non-neural layers and the final logits layer
    neural_layers = []
    for layer in all_layers:
        # Include q_net layers but exclude the final one (logits)
        if layer.startswith('q_net.'):
            # Extract layer number
            try:
                layer_num = int(layer.split('.')[1])
                # Exclude the highest numbered layer (assumed to be logits)
                # We'll determine this dynamically
                neural_layers.append((layer, layer_num))
            except (IndexError, ValueError):
                continue
    
    # Sort by layer number and exclude the highest (logits layer)
    neural_layers.sort(key=lambda x: x[1])
    
    if len(neural_layers) > 1:
        # Exclude the last layer (logits)
        neural_layers = neural_layers[:-1]
    
    layer_names = [layer[0] for layer in neural_layers]
    
    print(f"Found neural layers: {all_layers}")
    print(f"Training probes on layers: {layer_names}")
    
    return layer_names


def train_linear_probes(agent_path, clean_activations_file=None, corrupt_activations_file=None, 
                       output_dir=None, test_size=0.2, random_state=42):
    """
    Train linear probes on all non-logits layers.
    Includes both logistic regression (clean vs corrupt) and linear regression (logit prediction).
    
    Args:
        agent_path: Path to the agent directory
        clean_activations_file: Path to clean activations file (optional)
        corrupt_activations_file: Path to corrupt activations file (optional)
        output_dir: Output directory for saved models (optional)
        test_size: Proportion of data to use for testing
        random_state: Random seed for reproducibility
        
    Returns:
        dict: Dictionary of trained models and metrics for each layer and task
    """
    agent_path = Path(agent_path)
    
    # Default file paths if not provided
    if clean_activations_file is None:
        clean_activations_file = agent_path / "activation_logging" / "clean_activations_readable.json"
    
    if corrupt_activations_file is None:
        corrupt_activations_file = agent_path / "activation_logging" / "corrupted_activations_readable.json"
    
    if output_dir is None:
        output_dir = agent_path / "activation_probing"
    
    print(f"Loading activations...")
    print(f"Clean: {clean_activations_file}")
    print(f"Corrupt: {corrupt_activations_file}")
    
    # Load activations
    clean_activations = load_activations(clean_activations_file)
    corrupt_activations = load_activations(corrupt_activations_file)
    
    # Get layer names to train on (excluding logits)
    layer_names = get_layer_names_from_activations(clean_activations)
    
    if not layer_names:
        raise ValueError("No suitable layers found for training probes")
    
    results = {
        'logistic_regression': {},
        'linear_regression_clean': {},
        'linear_regression_corrupt': {}
    }
    
    # Train logistic regression probes (clean vs corrupt) for each layer
    print(f"\n{'='*80}")
    print(f"TRAINING LOGISTIC REGRESSION MODELS (Clean vs Corrupt Classification)")
    print(f"{'='*80}")
    
    for layer_name in layer_names:
        try:
            print(f"\n{'='*60}")
            print(f"Processing layer: {layer_name} (Logistic Regression)")
            print(f"{'='*60}")
            
            # Extract data for this layer
            X, y = extract_layer_data(clean_activations, corrupt_activations, layer_name)
            
            # Train the probe
            model, scaler, metrics = train_probe(X, y, layer_name, test_size, random_state)
            
            # Save the results to clean_vs_corrupt subfolder
            save_probe_results(model, scaler, metrics, output_dir, layer_name)
            
            results['logistic_regression'][layer_name] = {
                'model': model,
                'scaler': scaler,
                'metrics': metrics
            }
            
        except Exception as e:
            print(f"Error training logistic regression probe for layer {layer_name}: {e}")
            continue
    
    # Train linear regression models for clean activations
    print(f"\n{'='*80}")
    print(f"TRAINING LINEAR REGRESSION MODELS (Clean Activations -> Logit Prediction)")
    print(f"{'='*80}")
    
    for layer_name in layer_names:
        try:
            print(f"\n{'='*60}")
            print(f"Processing layer: {layer_name} (Linear Regression - Clean)")
            print(f"{'='*60}")
            
            # Extract clean activation data and logits
            X_clean, logits_clean = extract_logit_data(clean_activations, layer_name)
            
            # Compute target values
            top_logits, second_logits, margins = compute_logit_targets(logits_clean)
            
            # Train models for each target
            targets = [
                ('top_logit', top_logits),
                ('second_logit', second_logits),
                ('logit_margin', margins)
            ]
            
            layer_results = {}
            
            for target_name, target_values in targets:
                print(f"\n--- Training {target_name} model ---")
                
                # Train the model
                model, scaler, metrics = train_linear_regression(
                    X_clean, target_values, target_name, layer_name, test_size, random_state
                )
                
                # Save to appropriate subfolder
                target_output_dir = Path(output_dir) / target_name
                save_regression_results(model, scaler, metrics, target_output_dir, layer_name, target_name)
                
                layer_results[target_name] = {
                    'model': model,
                    'scaler': scaler,
                    'metrics': metrics
                }
            
            results['linear_regression_clean'][layer_name] = layer_results
            
        except Exception as e:
            print(f"Error training linear regression models for clean layer {layer_name}: {e}")
            continue
    
    # Train linear regression models for corrupt activations
    print(f"\n{'='*80}")
    print(f"TRAINING LINEAR REGRESSION MODELS (Corrupt Activations -> Logit Prediction)")
    print(f"{'='*80}")
    
    for layer_name in layer_names:
        try:
            print(f"\n{'='*60}")
            print(f"Processing layer: {layer_name} (Linear Regression - Corrupt)")
            print(f"{'='*60}")
            
            # Extract corrupt activation data and logits
            X_corrupt, logits_corrupt = extract_logit_data(corrupt_activations, layer_name)
            
            # Compute target values
            top_logits, second_logits, margins = compute_logit_targets(logits_corrupt)
            
            # Train models for each target
            targets = [
                ('top_logit', top_logits),
                ('second_logit', second_logits),
                ('logit_margin', margins)
            ]
            
            layer_results = {}
            
            for target_name, target_values in targets:
                print(f"\n--- Training {target_name} model (corrupt) ---")
                
                # Train the model
                model, scaler, metrics = train_linear_regression(
                    X_corrupt, target_values, f"{target_name}_corrupt", layer_name, test_size, random_state
                )
                
                # Save to appropriate subfolder with _corrupt suffix
                target_output_dir = Path(output_dir) / target_name
                save_regression_results(model, scaler, metrics, target_output_dir, layer_name, f"{target_name}_corrupt")
                
                layer_results[target_name] = {
                    'model': model,
                    'scaler': scaler,
                    'metrics': metrics
                }
            
            results['linear_regression_corrupt'][layer_name] = layer_results
            
        except Exception as e:
            print(f"Error training linear regression models for corrupt layer {layer_name}: {e}")
            continue
    
    # Save comprehensive summary of all results
    summary_file = Path(output_dir) / "comprehensive_training_summary.json"
    summary = {
        'logistic_regression': {
            layer: result['metrics'] for layer, result in results['logistic_regression'].items()
        },
        'linear_regression_clean': {
            layer: {target: result['metrics'] for target, result in layer_results.items()}
            for layer, layer_results in results['linear_regression_clean'].items()
        },
        'linear_regression_corrupt': {
            layer: {target: result['metrics'] for target, result in layer_results.items()}
            for layer, layer_results in results['linear_regression_corrupt'].items()
        }
    }
    
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Also save the old format summary for backwards compatibility
    old_summary_file = Path(output_dir) / "clean_vs_corrupt" / "probe_training_summary.json"
    old_summary = {
        layer: result['metrics'] for layer, result in results['logistic_regression'].items()
    }
    
    old_summary_file.parent.mkdir(parents=True, exist_ok=True)
    with open(old_summary_file, 'w') as f:
        json.dump(old_summary, f, indent=2)
    
    print(f"\n{'='*80}")
    print(f"ALL TRAINING COMPLETE!")
    print(f"{'='*80}")
    print(f"Results saved to: {output_dir}")
    print(f"Logistic regression models: {len(results['logistic_regression'])} layers")
    print(f"Linear regression models (clean): {len(results['linear_regression_clean'])} layers × 3 targets")
    print(f"Linear regression models (corrupt): {len(results['linear_regression_corrupt'])} layers × 3 targets")
    print(f"Comprehensive summary: {summary_file}")
    print(f"Backwards compatible summary: {old_summary_file}")
    print(f"{'='*80}")
    
    return results


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train linear probes on activation data.")
    
    parser.add_argument("--agent_path", required=True,
                        help="Path to the agent directory")
    
    parser.add_argument("--clean_activations", 
                        help="Path to clean activations JSON file")
    
    parser.add_argument("--corrupt_activations",
                        help="Path to corrupt activations JSON file")
    
    parser.add_argument("--output_dir",
                        help="Output directory for saved models")
    
    parser.add_argument("--test_size", type=float, default=0.2,
                        help="Proportion of data to use for testing (default: 0.2)")
    
    parser.add_argument("--random_state", type=int, default=42,
                        help="Random seed for reproducibility (default: 42)")
    
    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()
    
    try:
        results = train_linear_probes(
            agent_path=args.agent_path,
            clean_activations_file=args.clean_activations,
            corrupt_activations_file=args.corrupt_activations,
            output_dir=args.output_dir,
            test_size=args.test_size,
            random_state=args.random_state
        )
        
        print("\nTraining completed successfully!")
        
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def load_trained_probe(probe_dir, layer_name):
    """
    Load a trained probe model and scaler for a specific layer.
    
    Args:
        probe_dir: Directory containing the trained probe files
        layer_name: Name of the layer (e.g., 'q_net.0')
        
    Returns:
        tuple: (model, scaler, metrics) where model is the trained LogisticRegression,
               scaler is the fitted StandardScaler, and metrics contains training info
    """
    probe_dir = Path(probe_dir)
    
    # Try new structure first (clean_vs_corrupt subfolder)
    clean_vs_corrupt_dir = probe_dir / "clean_vs_corrupt"
    if clean_vs_corrupt_dir.exists():
        probe_dir = clean_vs_corrupt_dir
    
    # Load model
    model_file = probe_dir / f"{layer_name}_probe_model.pkl"
    if not model_file.exists():
        raise FileNotFoundError(f"Model file not found: {model_file}")
    
    with open(model_file, 'rb') as f:
        model = pickle.load(f)
    
    # Load scaler
    scaler_file = probe_dir / f"{layer_name}_probe_scaler.pkl"
    if not scaler_file.exists():
        raise FileNotFoundError(f"Scaler file not found: {scaler_file}")
    
    with open(scaler_file, 'rb') as f:
        scaler = pickle.load(f)
    
    # Load metrics
    metrics_file = probe_dir / f"{layer_name}_probe_metrics.json"
    metrics = None
    if metrics_file.exists():
        with open(metrics_file, 'r') as f:
            metrics = json.load(f)
    
    return model, scaler, metrics


def predict_with_probe(model, scaler, activation_data):
    """
    Use a trained probe to predict whether activation data is from clean or corrupt behavior.
    
    Args:
        model: Trained LogisticRegression model
        scaler: Fitted StandardScaler
        activation_data: Activation values for the layer (numpy array or list)
        
    Returns:
        tuple: (prediction, probability) where prediction is 0 (clean) or 1 (corrupt),
               and probability is the confidence of the prediction
    """
    # Convert to numpy array and reshape if needed
    if isinstance(activation_data, list):
        activation_data = np.array(activation_data)
    
    if activation_data.ndim == 1:
        activation_data = activation_data.reshape(1, -1)
    
    # Scale the data
    scaled_data = scaler.transform(activation_data)
    
    # Make prediction
    prediction = model.predict(scaled_data)[0]
    probabilities = model.predict_proba(scaled_data)[0]
    
    # Get confidence (probability of predicted class)
    confidence = probabilities[prediction]
    
    return prediction, confidence


def analyze_layer_importance(probe_dir):
    """
    Analyze and compare the importance of different layers based on probe performance.
    
    Args:
        probe_dir: Directory containing trained probe files
        
    Returns:
        dict: Summary of probe performance for each layer
    """
    probe_dir = Path(probe_dir)
    
    # Try new structure first (clean_vs_corrupt subfolder)
    summary_file = probe_dir / "clean_vs_corrupt" / "probe_training_summary.json"
    
    # Fall back to old structure for backwards compatibility
    if not summary_file.exists():
        summary_file = probe_dir / "probe_training_summary.json"
    
    if not summary_file.exists():
        raise FileNotFoundError(f"Summary file not found in either {probe_dir / 'clean_vs_corrupt' / 'probe_training_summary.json'} or {probe_dir / 'probe_training_summary.json'}")
    
    with open(summary_file, 'r') as f:
        summary = json.load(f)
    
    # Create analysis
    analysis = {}
    for layer_name, metrics in summary.items():
        # Get top neurons from new format or fall back to old format
        if 'neuron_rankings' in metrics:
            top_3_neurons = [ranking['neuron_index'] for ranking in metrics['neuron_rankings'][:3]]
            top_3_weights = [ranking['weight'] for ranking in metrics['neuron_rankings'][:3]]
            avg_weight = np.mean([ranking['weight'] for ranking in metrics['neuron_rankings']])
        else:
            # Fallback to old format
            top_3_neurons = metrics['top_features'][:3]
            top_3_weights = [metrics['feature_importance'][i] for i in top_3_neurons]
            avg_weight = np.mean(metrics['feature_importance'])
        
        analysis[layer_name] = {
            'test_accuracy': metrics['test_accuracy'],
            'train_accuracy': metrics['train_accuracy'],
            'overfitting': metrics['train_accuracy'] - metrics['test_accuracy'],
            'n_features': metrics['n_features'],
            'n_samples': metrics['n_samples'],
            'n_train_samples': metrics.get('n_train_samples', 'Unknown'),
            'n_test_samples': metrics.get('n_test_samples', 'Unknown'),
            'used_split': metrics.get('used_train_test_split', True),
            'dataset_warning': metrics.get('dataset_size_warning', False),
            'top_3_neurons': top_3_neurons,
            'top_3_weights': top_3_weights,
            'discriminative_power': avg_weight,
            'regularization': metrics.get('regularization', 'Not specified'),
            'solver_info': metrics.get('solver_info', 'Not specified')
        }
    
    # Sort by test accuracy
    sorted_layers = sorted(analysis.items(), key=lambda x: x[1]['test_accuracy'], reverse=True)
    
    print("Layer Performance Analysis (Logistic Regression - Clean vs Corrupt):")
    print("=" * 70)
    for layer_name, stats in sorted_layers:
        print(f"\nLayer: {layer_name}")
        print(f"  Test Accuracy: {stats['test_accuracy']:.4f}")
        print(f"  Train Accuracy: {stats['train_accuracy']:.4f}")
        print(f"  Overfitting: {stats['overfitting']:.4f}")
        print(f"  Total Samples: {stats['n_samples']} (Train: {stats['n_train_samples']}, Test: {stats['n_test_samples']})")
        if stats['dataset_warning']:
            print(f"  ⚠️  WARNING: Small dataset - results may not be reliable")
        if not stats['used_split']:
            print(f"  ⚠️  NOTE: No train/test split used (dataset too small)")
        print(f"  Features: {stats['n_features']}")
        print(f"  Regularization: {stats['regularization']}")
        if isinstance(stats['solver_info'], dict):
            solver_info = stats['solver_info']
            convergence_status = "✓ Converged" if solver_info.get('converged', False) else "✗ Did not converge"
            print(f"  Solver: {solver_info.get('solver', 'unknown')} ({convergence_status})")
            if solver_info.get('n_iter') is not None:
                print(f"  Iterations: {solver_info.get('n_iter')}/{solver_info.get('max_iter', 'unknown')}")
        else:
            print(f"  Solver: {stats['solver_info']}")
        print(f"  Top 3 Neurons: {stats['top_3_neurons']}")
        print(f"  Top 3 Weights: {[f'{w:.4f}' for w in stats['top_3_weights']]}")
        print(f"  Avg Weight: {stats['discriminative_power']:.4f}")
    
    return analysis


if __name__ == "__main__":
    main() 