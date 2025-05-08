#!/usr/bin/env python

"""
Configuration file for flexible spawn distributions.
Edit this file to configure how agents spawn during training.
"""

# Enable/disable flexible spawn distribution
USE_FLEXIBLE_SPAWN = True

# Choose which approach to use (only one should be True)
USE_FIXED_DISTRIBUTION = False
USE_STAGE_BASED_TRAINING = True
USE_CONTINUOUS_TRANSITION = False

# Basic spawn parameters
EXCLUDE_GOAL_ADJACENT = True       # Whether to avoid spawning adjacent to goal

# Fixed Distribution Configuration (used if USE_FIXED_DISTRIBUTION is True)
FIXED_DISTRIBUTION = {
    "type": "distance_goal",        # Options: "uniform", "poisson_goal", "gaussian_goal", "distance_goal"
    "params": {
        "favor_near": False,        # True = spawn near goal, False = spawn far from goal
        "lambda_param": 1.0,        # For poisson distributions (higher = steeper falloff)
        "sigma": 2.0,               # For gaussian distributions (higher = wider spread)
        "power": 1                  # For distance-based distributions (higher = more extreme)
    }
}

# Stage-Based Training Configuration (used if USE_STAGE_BASED_TRAINING is True)
# This creates distinct stages with different spawn distributions that change at specific points during training
STAGE_TRAINING_CONFIG = {
    "num_stages": 4,                # Number of training stages
    "distributions": [
        # Stage 1 (0-25% of training): Very close to goal (learn basic task)
        {"type": "poisson_goal", "params": {"lambda_param": 1.0, "favor_near": True}},
        
        # Stage 2 (25-50% of training): Medium distance from goal (learn pathing)
        {"type": "distance_goal", "params": {"favor_near": True, "power": 1}},
        
        # Stage 3 (50-75% of training): Farther from goal (learn long trajectories)
        {"type": "gaussian_goal", "params": {"sigma": 2.0, "favor_near": False}},
        
        # Stage 4 (75-100% of training): Anywhere in grid (master environment)
        {"type": "uniform"}
    ]
}

# Continuous Transition Configuration (used if USE_CONTINUOUS_TRANSITION is True)
# This creates a smooth transition from an initial to a target distribution
CONTINUOUS_TRANSITION_CONFIG = {
    # Initial distribution (at start of training)
    "initial_type": "poisson_goal",
    "initial_params": {
        "lambda_param": 1.0,
        "favor_near": True          # Start near goal
    },
    
    # Target distribution (at end of training)
    "target_type": "uniform",       # End with uniform distribution
    "target_params": {},
    
    # Transition control
    "rate": 1.0                     # 1.0 = linear transition, >1.0 = stay at start longer, <1.0 = reach target sooner
}

# Visualization Settings
VISUALIZE_SPAWN_DISTRIBUTIONS = True  # Whether to generate and save visualizations
VISUALIZE_FREQUENCY = 10000          # How often to visualize (timesteps) 