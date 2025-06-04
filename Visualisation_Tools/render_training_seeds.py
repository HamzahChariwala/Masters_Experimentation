#!/usr/bin/env python3
"""
Render seeds used for training and evaluation based on the standard configuration.
Training: base environment seed 12345, seed_increment 1
Evaluation: starting from seed 81102, seed_increment 1
"""

from env_plots import plot_lavacrossing_environment


def render_first_training_seeds():
    """Render the first 5 seeds used for training."""
    
    # Based on config analysis:
    # - environment: 12345 (base seed)
    # - seed_increment: 1
    base_seed = 12345
    seed_increment = 1
    num_seeds = 5
    
    print(f"Rendering first {num_seeds} training seeds...")
    print(f"Base seed: {base_seed}, increment: {seed_increment}")
    print()
    
    training_seeds = []
    for i in range(num_seeds):
        seed = base_seed + (i * seed_increment)
        training_seeds.append(seed)
        
        print(f"Rendering training seed {i+1}/5: {seed}")
        plot_lavacrossing_environment(seed)
        print()
    
    print("Completed rendering all training seeds!")
    print(f"Training seeds rendered: {training_seeds}")
    print()


def render_first_evaluation_seeds():
    """Render the first 5 seeds used for evaluation."""
    
    # Based on user specification:
    # - evaluation seeds start from 81102
    # - assuming same increment pattern as training
    base_seed = 81102
    seed_increment = 1
    num_seeds = 5
    
    print(f"Rendering first {num_seeds} evaluation seeds...")
    print(f"Base seed: {base_seed}, increment: {seed_increment}")
    print()
    
    evaluation_seeds = []
    for i in range(num_seeds):
        seed = base_seed + (i * seed_increment)
        evaluation_seeds.append(seed)
        
        print(f"Rendering evaluation seed {i+1}/5: {seed}")
        plot_lavacrossing_environment(seed)
        print()
    
    print("Completed rendering all evaluation seeds!")
    print(f"Evaluation seeds rendered: {evaluation_seeds}")
    print()


if __name__ == "__main__":
    print("=== Rendering Evaluation Seeds ===")
    render_first_evaluation_seeds()
    print("Check the raw_envs/ folder for the generated visualizations.") 