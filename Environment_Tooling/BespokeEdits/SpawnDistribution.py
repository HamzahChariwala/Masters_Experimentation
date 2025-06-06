import numpy as np
import gymnasium as gym
from gymnasium import spaces
import random
import matplotlib.pyplot as plt
import os
from stable_baselines3.common.callbacks import BaseCallback
import time
from pprint import pprint


class DistributionMap:
    """
    Generates and manages probability distributions over a grid.
    
    This class creates mappings where each cell in the grid is assigned a probability,
    and these probabilities can be sampled from to determine spawn locations.
    """
    
    def __init__(self, width, height):
        """
        Initialize a distribution map of the given dimensions.
        
        Parameters:
        ----------
        width : int
            Width of the grid
        height : int
            Height of the grid
        """
        self.width = width
        self.height = height
        self.probabilities = np.zeros((height, width))
        self.cumulative_mapping = None
    
    def uniform_distribution(self):
        """Apply a uniform distribution over all cells."""
        self.probabilities.fill(1.0 / (self.width * self.height))
        return self
    
    def poisson_from_point(self, center_x, center_y, lambda_param=2.0):
        """
        Create a Poisson-like distribution radiating from a specific point.
        
        The probability decreases exponentially as distance from the center increases.
        
        Parameters:
        ----------
        center_x : int
            X-coordinate of the center point
        center_y : int
            Y-coordinate of the center point
        lambda_param : float
            Controls how quickly probability falls off with distance (higher = steeper falloff)
        """
        # Initialize the probability map
        self.probabilities = np.zeros((self.height, self.width))
        
        # Compute distances from center and associated probabilities
        for y in range(self.height):
            for x in range(self.width):
                # Manhattan distance
                distance = abs(x - center_x) + abs(y - center_y)
                # Poisson-like probability (higher at the center, falls off with distance)
                prob = np.exp(-lambda_param * distance)
                self.probabilities[y, x] = prob
        
        # Normalize to make it a proper probability distribution
        self.probabilities /= np.sum(self.probabilities)
        return self
    
    def gaussian_from_point(self, center_x, center_y, sigma=1.0):
        """
        Create a Gaussian distribution centered at a specific point.
        
        Parameters:
        ----------
        center_x : int
            X-coordinate of the center point
        center_y : int
            Y-coordinate of the center point
        sigma : float
            Standard deviation of the Gaussian (controls the spread)
        """
        # Initialize the probability map
        self.probabilities = np.zeros((self.height, self.width))
        
        # Compute the Gaussian distribution
        for y in range(self.height):
            for x in range(self.width):
                # Euclidean distance squared
                dist_squared = (x - center_x)**2 + (y - center_y)**2
                # Gaussian probability
                prob = np.exp(-dist_squared / (2 * sigma**2))
                self.probabilities[y, x] = prob
        
        # Normalize to make it a proper probability distribution
        self.probabilities /= np.sum(self.probabilities)
        return self
    
    def distance_based_from_point(self, center_x, center_y, favor_near=True, power=2):
        """
        Create a distance-based distribution from a point.
        
        Parameters:
        ----------
        center_x : int
            X-coordinate of the center point
        center_y : int
            Y-coordinate of the center point
        favor_near : bool
            If True, cells closer to center have higher probability.
            If False, cells further from center have higher probability.
        power : int
            Controls how sharply probability changes with distance
        """
        # Initialize probability map
        self.probabilities = np.zeros((self.height, self.width))
        
        # Compute the maximum possible distance in the grid
        max_distance = abs(0 - self.width) + abs(0 - self.height)  # Manhattan distance
        
        # Compute distances and probabilities
        for y in range(self.height):
            for x in range(self.width):
                # Manhattan distance
                distance = abs(x - center_x) + abs(y - center_y)
                
                # Normalize distance to [0, 1]
                norm_distance = distance / max_distance
                
                if favor_near:
                    # Closer cells get higher probability
                    prob = 1 - (norm_distance ** power)
                else:
                    # Further cells get higher probability
                    prob = norm_distance ** power
                
                self.probabilities[y, x] = prob
        
        # Normalize to make it a proper probability distribution
        self.probabilities /= np.sum(self.probabilities)
        return self
    
    def multi_point_distribution(self, points, distribution_type="poisson", params=None):
        """
        Create a distribution based on multiple points.
        
        Parameters:
        ----------
        points : list of tuples
            List of (x, y) coordinates to use as centers
        distribution_type : str
            "poisson", "gaussian", or "distance"
        params : dict
            Parameters for the distribution (lambda_param, sigma, etc.)
        """
        if params is None:
            params = {}
        
        # Initialize temporary distributions
        temp_maps = []
        
        # Generate distribution from each point
        for x, y in points:
            temp_map = DistributionMap(self.width, self.height)
            
            if distribution_type == "poisson":
                lambda_param = params.get("lambda_param", 2.0)
                temp_map.poisson_from_point(x, y, lambda_param)
            elif distribution_type == "gaussian":
                sigma = params.get("sigma", 1.0)
                temp_map.gaussian_from_point(x, y, sigma)
            elif distribution_type == "distance":
                favor_near = params.get("favor_near", True)
                power = params.get("power", 2)
                temp_map.distance_based_from_point(x, y, favor_near, power)
            else:
                raise ValueError(f"Unknown distribution type: {distribution_type}")
            
            temp_maps.append(temp_map.probabilities)
        
        # Combine the probability maps (maximum value at each cell)
        self.probabilities = np.zeros((self.height, self.width))
        for temp_map in temp_maps:
            self.probabilities = np.maximum(self.probabilities, temp_map)
        
        # Normalize to make it a proper probability distribution
        self.probabilities /= np.sum(self.probabilities)
        return self
    
    def from_existing_distribution(self, distribution_array):
        """Load an existing numpy probability distribution."""
        if distribution_array.shape != (self.height, self.width):
            raise ValueError(f"Distribution shape {distribution_array.shape} does not match grid shape ({self.height}, {self.width})")
        
        self.probabilities = distribution_array.copy()
        # Ensure it's a valid distribution
        self.probabilities /= np.sum(self.probabilities)
        return self
    
    def mask_cells(self, mask):
        """
        Apply a binary mask to the probability distribution.
        
        Parameters:
        ----------
        mask : numpy.ndarray of shape (height, width)
            Binary mask where 1 indicates a valid cell and 0 indicates an invalid cell
        """
        # Apply the mask
        self.probabilities *= mask
        
        # Re-normalize if any valid cells remain
        if np.sum(self.probabilities) > 0:
            self.probabilities /= np.sum(self.probabilities)
        else:
            # If no valid cells remain, fall back to uniform over valid cells
            self.probabilities = mask.astype(float) / np.sum(mask) if np.sum(mask) > 0 else np.zeros_like(mask)
        
        return self
    
    def invert(self):
        """Invert the probability distribution."""
        # Find max probability to use for inversion formula
        max_prob = np.max(self.probabilities)
        
        # Invert the probabilities
        self.probabilities = max_prob - self.probabilities
        
        # Normalize
        self.probabilities /= np.sum(self.probabilities)
        return self
    
    def temporal_interpolation(self, target_distribution, progress):
        """
        Interpolate between the current distribution and a target distribution.
        
        Parameters:
        ----------
        target_distribution : DistributionMap
            The target distribution to interpolate toward
        progress : float
            Value between 0 and 1 indicating interpolation progress
            (0 = current distribution, 1 = target distribution)
        """
        if target_distribution.probabilities.shape != self.probabilities.shape:
            raise ValueError("Target distribution has different dimensions")
        
        # Clamp progress to [0, 1]
        progress = max(0, min(1, progress))
        
        # Store the mask of zero probability cells (cells that should remain zero)
        zero_mask = (self.probabilities == 0) & (target_distribution.probabilities == 0)
        
        # Interpolate between distributions
        self.probabilities = (1 - progress) * self.probabilities + progress * target_distribution.probabilities
        
        # Ensure zero probability cells remain zero
        self.probabilities[zero_mask] = 0
        
        # Re-normalize
        if np.sum(self.probabilities) > 0:
            self.probabilities /= np.sum(self.probabilities)
        
        return self
    
    def build_sampling_map(self):
        """Build a cumulative distribution for efficient sampling."""
        if np.sum(self.probabilities) == 0:
            # If all probabilities are zero, create a uniform distribution
            self.probabilities.fill(1.0 / (self.width * self.height))
        
        # Flatten and compute cumulative sum
        flat_probs = self.probabilities.flatten()
        self.cumulative_mapping = np.cumsum(flat_probs)
        
        return self
    
    def sample(self):
        """
        Sample a position from the distribution.
        
        Returns:
        -------
        tuple: (x, y) coordinates sampled from the distribution
        """
        if self.cumulative_mapping is None:
            self.build_sampling_map()
        
        # Sample from cumulative distribution
        r = np.random.random()
        idx = np.searchsorted(self.cumulative_mapping, r)
        
        # Convert flat index back to 2D coordinates
        y = idx // self.width
        x = idx % self.width
        
        return x, y
    
    def plot(self, title="Probability Distribution", show=True, save_path=None):
        """
        Plot the probability distribution as a heatmap.
        
        Parameters:
        ----------
        title : str
            Title for the plot
        show : bool
            Whether to display the plot
        save_path : str or None
            If provided, save the plot to this path
        """
        plt.figure(figsize=(10, 8))
        plt.imshow(self.probabilities, cmap='hot', interpolation='nearest')
        plt.colorbar(label='Probability')
        plt.title(title)
        plt.xlabel('X')
        plt.ylabel('Y')
        
        # Add grid lines
        plt.grid(True, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)
        
        # Add cell coordinates
        for i in range(self.width):
            for j in range(self.height):
                # Add text at center of each cell
                plt.text(i, j, f'({i},{j})', 
                         horizontalalignment='center',
                         verticalalignment='center',
                         color='black' if self.probabilities[j, i] < 0.5 else 'white',
                         fontsize=8)
        
        if save_path:
            plt.savefig(save_path)
        
        if show:
            plt.show()
        else:
            plt.close()
        
        return self

    def gaussian_2d(self, center, std, directional=False, angle=0):
        """
        Create a 2D Gaussian distribution with configurable center and standard deviations.
        
        Parameters:
        ----------
        center : list or tuple [x, y]
            Normalized center coordinates where (0,0) is top-left and (1,1) is bottom-right
        std : list or tuple [σx, σy]
            Normalized standard deviations in x and y directions or in the rotated directions
        directional : bool
            If True, the standard deviations are applied along rotated axes based on the angle
        angle : float
            Rotation angle in degrees (only used if directional=True)
            0 = standard x,y axes, 45 = top-left to bottom-right diagonal
        """
        # Initialize the probability map
        self.probabilities = np.zeros((self.height, self.width))
        
        # Convert normalized coordinates to grid coordinates
        center_x = center[0] * (self.width - 1)
        center_y = center[1] * (self.height - 1)
        
        # Convert normalized standard deviations to grid units
        # Use the larger dimension for scaling to maintain aspect ratio
        scale_factor = max(self.width, self.height)
        sigma_x = std[0] * scale_factor
        sigma_y = std[1] * scale_factor
        
        # Convert angle to radians if using directional Gaussian
        if directional:
            theta = np.radians(angle)
            cos_theta = np.cos(theta)
            sin_theta = np.sin(theta)
        
        # Compute the Gaussian distribution
        for y in range(self.height):
            for x in range(self.width):
                if directional:
                    # Rotate coordinates for directional Gaussian
                    # Shift by center, rotate, then compute distances along the rotated axes
                    x_shifted = x - center_x
                    y_shifted = y - center_y
                    
                    # Rotate coordinates
                    x_rot = x_shifted * cos_theta + y_shifted * sin_theta
                    y_rot = -x_shifted * sin_theta + y_shifted * cos_theta
                    
                    # Normalized distances in the rotated coordinate system
                    x_term = (x_rot / sigma_x) ** 2
                    y_term = (y_rot / sigma_y) ** 2
                else:
                    # Standard Gaussian - use regular x,y distances
                    x_term = ((x - center_x) / sigma_x) ** 2
                    y_term = ((y - center_y) / sigma_y) ** 2
                
                # Calculate probability using the 2D Gaussian formula
                exponent = -0.5 * (x_term + y_term)
                prob = np.exp(exponent)
                self.probabilities[y, x] = prob
        
        # Normalize to make it a proper probability distribution
        if np.sum(self.probabilities) > 0:
            self.probabilities /= np.sum(self.probabilities)
        return self

    def corners_distribution(self, corner_size=2):
        """
        Create a distribution focused on the corners of the grid.
        
        Parameters:
        ----------
        corner_size : int
            Size of each corner area in grid cells
        """
        # Initialize the probability map
        self.probabilities = np.zeros((self.height, self.width))
        
        # Define the corner regions
        corners = [
            # Top-left
            (0, 0, corner_size, corner_size),
            # Top-right
            (self.width - corner_size, 0, self.width, corner_size),
            # Bottom-left
            (0, self.height - corner_size, corner_size, self.height),
            # Bottom-right
            (self.width - corner_size, self.height - corner_size, self.width, self.height)
        ]
        
        # Set equal probability in each corner
        corner_prob = 1.0 / (4 * corner_size * corner_size)
        
        for x1, y1, x2, y2 in corners:
            for y in range(y1, y2):
                for x in range(x1, x2):
                    if 0 <= y < self.height and 0 <= x < self.width:
                        self.probabilities[y, x] = corner_prob
        
        # Normalize to make it a proper probability distribution
        if np.sum(self.probabilities) > 0:
            self.probabilities /= np.sum(self.probabilities)
        return self

    def border_distribution(self, border_width=1):
        """
        Create a distribution focused on the border of the grid.
        
        Parameters:
        ----------
        border_width : int
            Width of the border in grid cells
        """
        # Initialize the probability map
        self.probabilities = np.zeros((self.height, self.width))
        
        # Calculate total border cells
        total_border_cells = (2 * self.width * border_width + 
                              2 * (self.height - 2 * border_width) * border_width)
        
        # Set equal probability for all border cells
        border_prob = 1.0 / total_border_cells if total_border_cells > 0 else 0
        
        # Set border probabilities
        for y in range(self.height):
            for x in range(self.width):
                if (x < border_width or x >= self.width - border_width or 
                    y < border_width or y >= self.height - border_width):
                    self.probabilities[y, x] = border_prob
        
        # Normalize to make it a proper probability distribution
        if np.sum(self.probabilities) > 0:
            self.probabilities /= np.sum(self.probabilities)
        return self

    def composite_distribution(self, distributions):
        """
        Create a composite distribution by combining multiple weighted distributions.
        
        Parameters:
        ----------
        distributions : list of dicts
            Each dict contains:
              - type: Distribution type name
              - weight: Relative weight of this distribution (default: 1.0)
              - params: Parameters for this distribution
        """
        # Initialize zero probability map
        self.probabilities = np.zeros((self.height, self.width))
        
        total_weight = 0.0
        
        # Create a temporary distribution map for each component
        for dist_config in distributions:
            dist_type = dist_config["type"]
            dist_weight = dist_config.get("weight", 1.0)
            dist_params = dist_config.get("params", {})
            
            temp_map = DistributionMap(self.width, self.height)
            
            # Apply the appropriate distribution type
            if dist_type == "uniform":
                temp_map.uniform_distribution()
            elif dist_type == "gaussian_2d":
                center = dist_params.get("center", [0.5, 0.5])
                std = dist_params.get("std", [0.2, 0.2])
                directional = dist_params.get("directional", False)
                angle = dist_params.get("angle", 0)
                temp_map.gaussian_2d(center, std, directional, angle)
            elif dist_type == "corners":
                corner_size = dist_params.get("corner_size", 2)
                temp_map.corners_distribution(corner_size)
            elif dist_type == "border":
                border_width = dist_params.get("border_width", 1)
                temp_map.border_distribution(border_width)
            else:
                # Skip unknown distribution types
                continue
            
            # Add weighted contribution to the composite distribution
            self.probabilities += dist_weight * temp_map.probabilities
            total_weight += dist_weight
        
        # Normalize by total weight
        if total_weight > 0:
            self.probabilities /= total_weight
            
        # Ensure it's a valid probability distribution
        if np.sum(self.probabilities) > 0:
            self.probabilities /= np.sum(self.probabilities)
        return self


class FlexibleSpawnWrapper(gym.Wrapper):
    """
    A wrapper that provides flexible spawn distribution functionality for grid environments.
    
    This wrapper allows spawning the agent according to various probability distributions
    over the grid cells, and supports temporal changes to these distributions during training.
    """
    
    def __init__(self, env, distribution_type="uniform", total_timesteps=None, 
                 exclude_occupied=True, exclude_goal_adjacent=False,
                 distribution_params=None, temporal_transition=None,
                 stage_based_training=None):
        """
        Initialize the wrapper with the specified distribution settings.
        
        Parameters:
        ----------
        env : gym.Env
            The environment to wrap
        distribution_type : str
            Type of distribution to use: "uniform", "poisson_goal", "poisson_lava", 
            "distance_goal", "gaussian_goal", etc.
        total_timesteps : int
            Total number of timesteps expected in training (for temporal transitions)
        exclude_occupied : bool
            If True, avoid spawning on non-empty cells
        exclude_goal_adjacent : bool
            If True, avoid spawning adjacent to the goal
        distribution_params : dict
            Parameters for the chosen distribution
        temporal_transition : dict
            Configuration for continuous distribution changes over time
        stage_based_training : dict
            Configuration for stage-based distribution changes:
            {
                "num_stages": int,  # Number of stages
                "distributions": [  # List of distribution configurations
                    {
                        "type": str,  # Distribution type
                        "params": dict,  # Distribution parameters
                        "relative_duration": float,  # Relative duration of this stage
                    },
                    ...
                ],
                "curriculum_proportion": float,  # Proportion of total timesteps to use for curriculum
                "smooth_transitions": {
                    "enabled": bool,  # Whether to use smooth transitions
                    "transition_proportion": float,  # Proportion of each stage for transition
                }
            }
        """
        super().__init__(env)
        self.distribution_type = distribution_type
        self.total_timesteps = total_timesteps
        self.exclude_occupied = exclude_occupied
        self.exclude_goal_adjacent = exclude_goal_adjacent
        self.distribution_params = distribution_params or {}
        self.temporal_transition = temporal_transition
        self.stage_based_training = stage_based_training
        self.verbose = 1  # Set verbosity level for logging
        
        # Initialize timestep counter
        self.timestep = 0
        
        # Stage-based tracking
        self.current_stage = 0
        self.next_stage_transition = 0
        self.curriculum_end = None  # Timestep when the curriculum ends
        
        # Calculate curriculum-based timings if applicable
        if stage_based_training and total_timesteps:
            # Get curriculum proportion (default to 1.0 if not specified)
            curriculum_proportion = stage_based_training.get('curriculum_proportion', 1.0)
            
            # Calculate when the curriculum should end
            self.curriculum_end = int(total_timesteps * curriculum_proportion)
            
            if 'distributions' in stage_based_training:
                # Calculate stage durations based on relative durations
                distributions = stage_based_training['distributions']
                total_relative_duration = stage_based_training.get('total_relative_duration', None)
                
                # If total_relative_duration wasn't calculated, do it now
                if total_relative_duration is None:
                    total_relative_duration = 0
                    for dist in distributions:
                        total_relative_duration += dist.get('relative_duration', 1.0)
                
                # Calculate the timestamps for stage transitions
                self.stage_durations = []
                self.stage_transitions = [0]  # Start at timestep 0
                
                for i, dist in enumerate(distributions):
                    relative_duration = dist.get('relative_duration', 1.0)
                    # Calculate stage duration as proportion of curriculum duration
                    stage_duration = int((relative_duration / total_relative_duration) * self.curriculum_end)
                    self.stage_durations.append(stage_duration)
                    # Add this duration to the previous transition point
                    next_transition = self.stage_transitions[-1] + stage_duration
                    self.stage_transitions.append(next_transition)
                
                # Initialize the next transition point
                if len(self.stage_transitions) > 1:
                    self.next_stage_transition = self.stage_transitions[1]
                    
                if self.verbose > 0:
                    print(f"Curriculum transitions will occur at timesteps: {self.stage_transitions[1:]}")
                    print(f"Curriculum will end at timestep: {self.curriculum_end}")
                
        # Placeholder for distribution maps
        self.current_distribution = None
        self.target_distribution = None
        self.valid_cells_mask = None
        
        # For tracking and visualization (disabled for performance)
        self.distribution_history = None  # Changed from [] to None to save memory
    
    def reset(self, **kwargs):
        """Reset the environment and potentially respawn the agent."""
        # First reset the environment
        obs, info = self.env.reset(**kwargs)
        
        # Get grid properties
        grid = self.unwrapped.grid
        width, height = grid.width, grid.height
        
        # Create valid cells mask if not already created
        if self.valid_cells_mask is None or (hasattr(self.unwrapped, 'grid_changed') and self.unwrapped.grid_changed):
            self.valid_cells_mask = np.ones((height, width), dtype=int)
            
            # Find goal position
            self.goal_pos = None
            
            # Find all occupied cells and goal position
            for i in range(width):
                for j in range(height):
                    cell = grid.get(i, j)
                    
                    # Track goal position
                    if cell and cell.type == 'goal':
                        self.goal_pos = (i, j)
                    
                    # Mark occupied cells as invalid
                    if self.exclude_occupied and cell is not None and cell.type not in ['empty', 'goal']:
                        self.valid_cells_mask[j, i] = 0
                    
                    # Also mark lava cells as invalid for spawning
                    if cell and cell.type == 'lava':
                        self.valid_cells_mask[j, i] = 0
            
            # Always mark the goal position as invalid for spawning
            if self.goal_pos:
                self.valid_cells_mask[self.goal_pos[1], self.goal_pos[0]] = 0
            
            # Mark cells adjacent to goal as invalid if requested
            if self.exclude_goal_adjacent and self.goal_pos:
                goal_x, goal_y = self.goal_pos
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        nx, ny = goal_x + dx, goal_y + dy
                        if 0 <= nx < width and 0 <= ny < height:
                            self.valid_cells_mask[ny, nx] = 0
            
            # Create the distributions if they don't exist
            self._init_distributions(width, height)
        
        # Execute spawning based on current distribution
        if np.sum(self.valid_cells_mask) > 0:
            # Sample a position from the distribution
            x, y = self.current_distribution.sample()
            
            # Set agent position
            self.unwrapped.agent_pos = (x, y)
            
            # Update agent's position in the grid
            self.unwrapped.grid.set(x, y, None)
            
            # Recalculate the observation
            if hasattr(self.unwrapped, 'gen_obs'):
                # MiniGrid environments use gen_obs()
                obs = self.unwrapped.gen_obs()
            elif hasattr(self.unwrapped, '_get_obs'):
                # Some environments use _get_obs()
                obs = self.unwrapped._get_obs()
        
        return obs, info
    
    def _init_distributions(self, width, height):
        """Initialize the probability distributions based on settings."""
        # Create initial distribution
        self.current_distribution = DistributionMap(width, height)
        
        # For stage-based training, initialize with the first stage
        if self.stage_based_training and 'distributions' in self.stage_based_training and self.stage_based_training['distributions']:
            # Use first distribution from stages
            first_stage = self.stage_based_training['distributions'][0]
            self._apply_distribution(first_stage['type'], first_stage.get('params', {}))
        else:
            # Apply the appropriate distribution based on distribution_type
            self._apply_distribution(self.distribution_type, self.distribution_params)
        
        # Always ensure goal position has zero probability
        if self.goal_pos:
            goal_mask = np.ones((height, width), dtype=int)
            goal_mask[self.goal_pos[1], self.goal_pos[0]] = 0
            self.current_distribution.mask_cells(goal_mask)
        
        # Apply valid cells mask (which should already exclude the goal)
        self.current_distribution.mask_cells(self.valid_cells_mask)
        
        # Create target distribution if temporal transition is enabled
        if self.temporal_transition and self.total_timesteps:
            # Get target distribution type
            target_type = self.temporal_transition.get("target_type", "uniform")
            target_params = self.temporal_transition.get("target_params", {})
            
            # Create target distribution
            self.target_distribution = DistributionMap(width, height)
            
            if target_type == "uniform":
                self.target_distribution.uniform_distribution()
            
            elif target_type == "poisson_goal" and self.goal_pos:
                lambda_param = target_params.get("lambda_param", 2.0)
                favor_near = target_params.get("favor_near", False)  # Default to far for target
                
                self.target_distribution.poisson_from_point(
                    self.goal_pos[0], self.goal_pos[1], lambda_param
                )
                
                if not favor_near:
                    self.target_distribution.invert()
            
            elif target_type == "gaussian_goal" and self.goal_pos:
                sigma = target_params.get("sigma", 1.0)
                favor_near = target_params.get("favor_near", False)  # Default to far for target
                
                self.target_distribution.gaussian_from_point(
                    self.goal_pos[0], self.goal_pos[1], sigma
                )
                
                if not favor_near:
                    self.target_distribution.invert()
            
            elif target_type == "distance_goal" and self.goal_pos:
                favor_near = target_params.get("favor_near", False)  # Default to far for target
                power = target_params.get("power", 2)
                
                self.target_distribution.distance_based_from_point(
                    self.goal_pos[0], self.goal_pos[1], favor_near, power
                )
            
            else:
                # Default target is uniform
                self.target_distribution.uniform_distribution()
            
            # Always ensure goal has zero probability in target distribution too
            if self.goal_pos:
                goal_mask = np.ones((height, width), dtype=int)
                goal_mask[self.goal_pos[1], self.goal_pos[0]] = 0
                self.target_distribution.mask_cells(goal_mask)
            
            # Apply valid cells mask to target
            self.target_distribution.mask_cells(self.valid_cells_mask)
        
        # Prepare the distribution for sampling
        self.current_distribution.build_sampling_map()
        
        # Disabled history collection to improve performance
        # self.distribution_history = [(0, initial_dist)]
    
    def _apply_distribution(self, dist_type, dist_params):
        """Apply a specific distribution type with given parameters."""
        params = dist_params or {}
        
        if dist_type == "uniform":
            self.current_distribution.uniform_distribution()
        
        elif dist_type == "gaussian_2d":
            center = params.get("center", [1.0, 1.0])  # Default to goal position (bottom-right)
            std = params.get("std", [0.2, 0.2])
            directional = params.get("directional", False)
            angle = params.get("angle", 0)
            
            # Use the new 2D Gaussian distribution
            self.current_distribution.gaussian_2d(center, std, directional, angle)
        
        elif dist_type == "corners":
            corner_size = params.get("corner_size", 2)
            self.current_distribution.corners_distribution(corner_size)
        
        elif dist_type == "border":
            border_width = params.get("border_width", 1)
            self.current_distribution.border_distribution(border_width)
        
        elif dist_type == "composite":
            distributions = params.get("distributions", [])
            self.current_distribution.composite_distribution(distributions)
        
        elif dist_type == "poisson_goal" and self.goal_pos:
            lambda_param = params.get("lambda_param", 2.0)
            favor_near = params.get("favor_near", True)
            
            # Create distribution centered on goal
            self.current_distribution.poisson_from_point(
                self.goal_pos[0], self.goal_pos[1], lambda_param
            )
            
            # Invert if we want to favor positions far from goal
            if not favor_near:
                self.current_distribution.invert()
        
        elif dist_type == "gaussian_goal" and self.goal_pos:
            sigma = params.get("sigma", 1.0)
            favor_near = params.get("favor_near", True)
            
            # Create Gaussian distribution centered on goal
            self.current_distribution.gaussian_from_point(
                self.goal_pos[0], self.goal_pos[1], sigma
            )
            
            # Invert if we want to favor positions far from goal
            if not favor_near:
                self.current_distribution.invert()
        
        elif dist_type == "distance_goal" and self.goal_pos:
            favor_near = params.get("favor_near", True)
            power = params.get("power", 2)
            
            # Create distance-based distribution centered on goal
            self.current_distribution.distance_based_from_point(
                self.goal_pos[0], self.goal_pos[1], favor_near, power
            )
        
        else:
            # Default to uniform distribution
            self.current_distribution.uniform_distribution()
    
    def step(self, action):
        """
        Take a step and update distribution if needed.
        
        Parameters:
        ----------
        action : int
            The action to take
        
        Returns:
        -------
        obs : object
            The observation
        reward : float
            The reward
        done : bool
            Whether the episode is done
        info : dict
            Additional information
        """
        # Take the step in the environment
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Update timestep counter
        self.timestep += 1
        
        # Handle distribution updates based on timesteps
        if self.stage_based_training and self.total_timesteps:
            # Check if we've reached the end of the curriculum
            if self.curriculum_end is not None and self.timestep >= self.curriculum_end:
                # If we haven't already switched to uniform after curriculum
                if self.current_stage != -1:
                    # Switch to uniform distribution after curriculum is done
                    self._apply_distribution("uniform", {})
                    
                    # Apply masks to ensure goal and invalid cells have zero probability
                    if self.goal_pos:
                        goal_mask = np.ones_like(self.valid_cells_mask)
                        goal_mask[self.goal_pos[1], self.goal_pos[0]] = 0
                        self.current_distribution.mask_cells(goal_mask)
                    
                    self.current_distribution.mask_cells(self.valid_cells_mask)
                    self.current_distribution.build_sampling_map()
                    
                    # Mark as post-curriculum by setting stage to -1
                    self.current_stage = -1
                    
                    if self.verbose > 0:
                        print(f"Curriculum completed at timestep {self.timestep}. Switching to uniform distribution.")
                
                # No need to process further distribution updates
                return obs, reward, terminated, truncated, info
            
            # Get smooth transition settings if available
            smooth_transitions = self.stage_based_training.get('smooth_transitions', {'enabled': False})
            smooth_enabled = smooth_transitions.get('enabled', False)
            
            # Calculate transition duration as a proportion of stage duration
            transition_proportion = smooth_transitions.get('transition_proportion', 0.2)
            transition_rate = smooth_transitions.get('transition_rate', 'linear')
            
            # Check if we're still within the valid stages
            if self.current_stage < len(self.stage_based_training['distributions']) - 1:
                # Get current stage duration and transition points
                current_stage_end = self.stage_transitions[self.current_stage + 1]
                
                # Calculate transition parameters
                if smooth_enabled:
                    stage_duration = self.stage_durations[self.current_stage]
                    transition_duration = int(stage_duration * transition_proportion)
                    transition_start = current_stage_end - transition_duration
                    
                    # If we're in the transition period between stages
                    if self.timestep >= transition_start and self.timestep < current_stage_end:
                        # Calculate progress through the transition (0.0 to 1.0)
                        progress = (self.timestep - transition_start) / transition_duration
                        
                        # Apply different transition rates
                        if transition_rate == 'exponential':
                            progress = progress ** 2  # Exponential transition
                        elif transition_rate == 'sigmoid':
                            # Sigmoid function to create an S-shaped transition
                            progress = 1 / (1 + np.exp(-10 * (progress - 0.5)))
                        # 'linear' is the default, so we don't modify progress in that case
                        
                        # Get current and next stage configurations
                        current_stage_config = self.stage_based_training['distributions'][self.current_stage]
                        next_stage_config = self.stage_based_training['distributions'][self.current_stage + 1]
                        
                        # Create temporary distribution maps for interpolation
                        current_dist_map = DistributionMap(self.current_distribution.width, self.current_distribution.height)
                        next_dist_map = DistributionMap(self.current_distribution.width, self.current_distribution.height)
                        
                        # Apply distributions
                        self._apply_distribution_to_map(current_stage_config['type'], 
                                                    current_stage_config.get('params', {}),
                                                    current_dist_map)
                        
                        self._apply_distribution_to_map(next_stage_config['type'],
                                                    next_stage_config.get('params', {}),
                                                    next_dist_map)
                        
                        # Apply masks to both distributions
                        if self.goal_pos:
                            goal_mask = np.ones_like(self.valid_cells_mask)
                            goal_mask[self.goal_pos[1], self.goal_pos[0]] = 0
                            current_dist_map.mask_cells(goal_mask)
                            next_dist_map.mask_cells(goal_mask)
                        
                        current_dist_map.mask_cells(self.valid_cells_mask)
                        next_dist_map.mask_cells(self.valid_cells_mask)
                        
                        # Interpolate between the two distributions
                        self.current_distribution.from_existing_distribution(current_dist_map.probabilities)
                        self.current_distribution.temporal_interpolation(next_dist_map, progress)
                        self.current_distribution.build_sampling_map()
                    
                # If we've completed the transition to the next stage
                elif self.timestep >= current_stage_end:
                    # Move to next stage
                    self.current_stage += 1
                    
                    # Apply the distribution for the new stage
                    next_stage_config = self.stage_based_training['distributions'][self.current_stage]
                    self._apply_distribution(next_stage_config['type'], next_stage_config.get('params', {}))
                    
                    # Apply masks to ensure goal and invalid cells have zero probability
                    if self.goal_pos:
                        goal_mask = np.ones_like(self.valid_cells_mask)
                        goal_mask[self.goal_pos[1], self.goal_pos[0]] = 0
                        self.current_distribution.mask_cells(goal_mask)
                    
                    self.current_distribution.mask_cells(self.valid_cells_mask)
                    self.current_distribution.build_sampling_map()
                    
                    if self.verbose > 0:
                        print(f"Transitioned to stage {self.current_stage + 1}/{len(self.stage_based_training['distributions'])}")
                        print(f"Distribution: {next_stage_config['type']}")
                        if 'description' in next_stage_config:
                            print(f"Description: {next_stage_config['description']}")
                        print(f"Relative duration: {next_stage_config.get('relative_duration', 1.0)}")
        
        # Handle continuous transition
        elif self.temporal_transition and self.target_distribution and self.total_timesteps:
            # Calculate transition progress (0 to 1)
            rate = self.temporal_transition.get("rate", 1.0)  # Default to linear
            progress = min(1.0, (self.timestep / self.total_timesteps) ** rate)
            
            # Update distribution through interpolation
            self.current_distribution.temporal_interpolation(self.target_distribution, progress)
            self.current_distribution.build_sampling_map()
        
        return obs, reward, terminated, truncated, info
    
    def _apply_distribution_to_map(self, dist_type, dist_params, dist_map):
        """Apply a specific distribution type with given parameters to a provided distribution map."""
        params = dist_params or {}
        
        if dist_type == "uniform":
            dist_map.uniform_distribution()
        
        elif dist_type == "gaussian_2d":
            center = params.get("center", [1.0, 1.0])  # Default to goal position (bottom-right)
            std = params.get("std", [0.2, 0.2])
            directional = params.get("directional", False)
            angle = params.get("angle", 0)
            
            # Use the new 2D Gaussian distribution
            dist_map.gaussian_2d(center, std, directional, angle)
        
        elif dist_type == "corners":
            corner_size = params.get("corner_size", 2)
            dist_map.corners_distribution(corner_size)
        
        elif dist_type == "border":
            border_width = params.get("border_width", 1)
            dist_map.border_distribution(border_width)
        
        elif dist_type == "composite":
            distributions = params.get("distributions", [])
            dist_map.composite_distribution(distributions)
        
        elif dist_type == "poisson_goal" and self.goal_pos:
            lambda_param = params.get("lambda_param", 2.0)
            favor_near = params.get("favor_near", True)
            
            # Create distribution centered on goal
            dist_map.poisson_from_point(
                self.goal_pos[0], self.goal_pos[1], lambda_param
            )
            
            # Invert if we want to favor positions far from goal
            if not favor_near:
                dist_map.invert()
        
        elif dist_type == "gaussian_goal" and self.goal_pos:
            sigma = params.get("sigma", 1.0)
            favor_near = params.get("favor_near", True)
            
            # Create Gaussian distribution centered on goal
            dist_map.gaussian_from_point(
                self.goal_pos[0], self.goal_pos[1], sigma
            )
            
            # Invert if we want to favor positions far from goal
            if not favor_near:
                dist_map.invert()
        
        elif dist_type == "distance_goal" and self.goal_pos:
            favor_near = params.get("favor_near", True)
            power = params.get("power", 2)
            
            # Create distance-based distribution centered on goal
            dist_map.distance_based_from_point(
                self.goal_pos[0], self.goal_pos[1], favor_near, power
            )
        
        else:
            # Default to uniform distribution
            dist_map.uniform_distribution()
    
    def get_current_distribution_probabilities(self):
        """
        Get the current probability distribution array.
        
        Returns:
        -------
        numpy.ndarray
            2D array of probabilities
        """
        if self.current_distribution:
            return self.current_distribution.probabilities.copy()
        return None
    
    def visualize_distribution(self, title=None, save_path=None):
        """Visualize the current spawn distribution."""
        if self.current_distribution:
            title = title or f"Spawn Distribution (Timestep {self.timestep})"
            self.current_distribution.plot(title=title, save_path=save_path)
    
    def get_distribution_history(self):
        """
        Get the distribution history (for analysis only).
        If history collection is disabled, returns None.
        
        Returns:
        -------
        list or None
            List of (timestep, distribution) tuples, or None if history is disabled
        """
        return self.distribution_history
    
    def reset_timestep(self):
        """Reset the timestep counter and return to initial distribution."""
        self.timestep = 0
        self.current_stage = 0
        self.next_stage_transition = self.stage_timesteps if self.stage_based_training else 0
        
        # Reset distribution to initial state
        if self.stage_based_training and 'distributions' in self.stage_based_training:
            first_stage = self.stage_based_training['distributions'][0]
            self._apply_distribution(first_stage['type'], first_stage.get('params', {}))
            self.current_distribution.mask_cells(self.valid_cells_mask)
            self.current_distribution.build_sampling_map()


class SpawnDistributionCallback(BaseCallback):
    def __init__(self, env, save_dir=None, freq=10000, verbose=0):
        super(SpawnDistributionCallback, self).__init__(verbose)
        self.env = env
        self.save_dir = save_dir
        self.freq = freq
        
        # Create the save directory if it doesn't exist
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
    
    def _on_step(self) -> bool:
        # Record distribution at specific frequencies
        if self.num_timesteps % self.freq == 0:
            # Get the unwrapped environment
            if hasattr(self.env, 'venv'):
                # For vectorized environments, we need to get to a single env
                if hasattr(self.env.venv, 'envs'):
                    # DummyVecEnv case
                    env = self.env.venv.envs[0]
                else:
                    # Abstract VecEnv case - may need to be adapted
                    env = self.env.venv
            else:
                env = self.env
            
            # Try to find the FlexibleSpawnWrapper
            while env and not isinstance(env, FlexibleSpawnWrapper):
                if hasattr(env, 'env'):
                    env = env.env
                else:
                    # Couldn't find the wrapper
                    return True
            
            # If we found the wrapper, visualize the distribution
            if isinstance(env, FlexibleSpawnWrapper):
                if self.save_dir:
                    # Save the visualization to file
                    filename = f"spawn_dist_{self.num_timesteps}.png"
                    save_path = os.path.join(self.save_dir, filename)
                    env.visualize_distribution(
                        title=f"Spawn Distribution at Timestep {self.num_timesteps}",
                        save_path=save_path
                    )
                else:
                    # Just display the visualization (not ideal for headless operation)
                    env.visualize_distribution(
                        title=f"Spawn Distribution at Timestep {self.num_timesteps}"
                    )
        
        return True


class EnhancedSpawnDistributionCallback(BaseCallback):
    def __init__(self, env, save_dir=None, freq=10000, verbose=0):
        super(EnhancedSpawnDistributionCallback, self).__init__(verbose)
        self.env = env
        self.save_dir = save_dir
        self.freq = freq
        
        # Create the save directory if it doesn't exist
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        
        self.last_visualization = 0
    
    def _on_step(self) -> bool:
        # Only check at the specified frequency to save computation
        if self.num_timesteps - self.last_visualization >= self.freq:
            self.last_visualization = self.num_timesteps
            
            # Find all FlexibleSpawnWrapper instances in vectorized environments
            if hasattr(self.env, 'venv'):
                # For vectorized environments
                spawn_wrappers = []
                
                if hasattr(self.env.venv, 'envs'):
                    # DummyVecEnv case
                    for env in self.env.venv.envs:
                        wrapper = self._find_spawn_wrapper(env)
                        if wrapper:
                            spawn_wrappers.append(wrapper)
                else:
                    # Abstract VecEnv case
                    wrapper = self._find_spawn_wrapper(self.env.venv)
                    if wrapper:
                        spawn_wrappers.append(wrapper)
            else:
                # Non-vectorized environment
                wrapper = self._find_spawn_wrapper(self.env)
                spawn_wrappers = [wrapper] if wrapper else []
            
            # Visualize distributions for all found wrappers
            for i, wrapper in enumerate(spawn_wrappers):
                if self.save_dir:
                    # Save the visualization to file
                    filename = f"spawn_dist_{self.num_timesteps}_env{i}.png"
                    save_path = os.path.join(self.save_dir, filename)
                    wrapper.visualize_distribution(
                        title=f"Env {i} Spawn Distribution at Timestep {self.num_timesteps}",
                        save_path=save_path
                    )
                else:
                    # Just display the visualization
                    wrapper.visualize_distribution(
                        title=f"Env {i} Spawn Distribution at Timestep {self.num_timesteps}"
                    )
        
        return True
    
    def _find_spawn_wrapper(self, env):
        """Recursively find the FlexibleSpawnWrapper in a nested environment."""
        if isinstance(env, FlexibleSpawnWrapper):
            return env
        
        if hasattr(env, 'env'):
            return self._find_spawn_wrapper(env.env)
        
        return None


def generate_final_visualizations(env, save_dir=None):
    """
    Generate summary visualizations of spawn distribution evolution.
    
    Parameters:
    ----------
    env : gym.Env
        The environment containing FlexibleSpawnWrapper
    save_dir : str
        Directory to save visualizations
    """
    # Create the save directory if it doesn't exist
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    
    # Find the FlexibleSpawnWrapper
    wrapper = None
    while env and not isinstance(env, FlexibleSpawnWrapper):
        if hasattr(env, 'env'):
            env = env.env
        else:
            # Couldn't find the wrapper
            if wrapper is None:
                print("Warning: Could not find FlexibleSpawnWrapper")
            return
    
    if not isinstance(env, FlexibleSpawnWrapper):
        print("Warning: Could not find FlexibleSpawnWrapper")
        return
    
    wrapper = env
    
    # Check if history is available (may be disabled for performance)
    history = wrapper.get_distribution_history()
    if history is None or not history:
        print("Warning: Distribution history is not available (disabled for performance)")
        
        # Just visualize the current distribution
        if save_dir:
            save_path = os.path.join(save_dir, "final_distribution.png")
            wrapper.visualize_distribution(
                title="Final Spawn Distribution",
                save_path=save_path
            )
        else:
            wrapper.visualize_distribution(title="Final Spawn Distribution")
        return
    
    # Create a figure for the visualization timeline
    fig, axes = plt.subplots(1, len(history), figsize=(15, 5))
    if len(history) == 1:
        axes = [axes]  # Make it indexable if only one subplot
    
    # Plot each distribution
    for i, (timestep, dist) in enumerate(history):
        ax = axes[i]
        im = ax.imshow(dist, cmap='hot', interpolation='nearest')
        ax.set_title(f"t={timestep}")
        ax.axis('off')
    
    # Add a colorbar
    fig.colorbar(im, ax=axes, shrink=0.8, label='Probability')
    
    # Set the main title
    fig.suptitle("Spawn Distribution Evolution", fontsize=16)
    
    # Save the figure if a directory is provided
    if save_dir:
        save_path = os.path.join(save_dir, "distribution_evolution.png")
        plt.savefig(save_path)
        print(f"Saved distribution evolution visualization to {save_path}")
    
    # Show the figure
    plt.tight_layout()
    plt.show()


def print_numeric_distribution(grid, goal_pos=None, lava_positions=None, wall_positions=None, title=None, summary=True):
    """
    Print the actual numeric probability distribution with formatting.
    
    Parameters:
    ----------
    grid : numpy.ndarray
        Probability distribution array
    goal_pos : tuple (x, y) or None
        Position of the goal (if known)
    lava_positions : list of tuples (x, y) or None
        Positions of lava cells (if known)
    wall_positions : list of tuples (x, y) or None
        Positions of wall cells (if known)
    title : str or None
        Optional title to display 
    summary : bool
        Whether to print summary statistics
    """
    if title:
        print(f"\n=== {title} ===")
    
    height, width = grid.shape
    
    if lava_positions is None:
        lava_positions = []
    
    if wall_positions is None:
        wall_positions = []
    
    print("\n  Probability Distribution Array:")
    
    # Define column width and format strings for better alignment
    cell_width = 8  # Fixed width for all cells
    col_width = 9   # Total column width including spacing
    num_format = "{:.2e}"  # Scientific notation for all numbers to save space
    
    # Print column headers with better spacing
    header = "     |"  # Extra space for row numbers
    for x in range(width):
        header += f"{x:^{cell_width}}|"
    print(header)
    
    # Print horizontal line
    print("-----|" + "-" * (cell_width * width + width))
    
    # Format the probabilities with special symbols for goal, lava, and walls
    for y in range(height):
        row = f"{y:3d}  |"
        for x in range(width):
            if goal_pos and (x, y) == goal_pos:
                # Goal position - center aligned
                row += f"{'GOAL':^{cell_width}}|"
            elif (x, y) in lava_positions:
                # Lava position - center aligned
                row += f"{'LAVA':^{cell_width}}|"
            elif (x, y) in wall_positions:
                # Wall position - center aligned
                row += f"{'WALL':^{cell_width}}|"
            else:
                # Regular cell with probability value - right aligned
                prob = grid[y, x]
                row += f"{num_format.format(prob):>{cell_width}}|"
        print(row)
    
    # Print legend
    print("\n  Legend:")
    print("  - GOAL: Goal position (zero probability)")
    print("  - LAVA: Lava cell (zero probability)")
    print("  - WALL: Wall cell (zero probability)")
    print("  - Values represent spawn probabilities (sum to 1.0)")
    
    # Print summary statistics
    if summary:
        nonzero_cells = np.count_nonzero(grid)
        total_cells = width * height
        
        # Calculate the number of obstacle cells (walls, lava, goal)
        obstacle_cells = len(wall_positions) + len(lava_positions)
        if goal_pos:
            obstacle_cells += 1
            
        print(f"\n  Summary:")
        print(f"  - Valid spawn cells: {nonzero_cells} of {total_cells} ({nonzero_cells/total_cells:.1%})")
        print(f"  - Obstacles: {obstacle_cells} cells (goal: 1, lava: {len(lava_positions)}, walls: {len(wall_positions)})")
        
        if nonzero_cells > 0:
            print(f"  - Highest probability: {np.max(grid):.5f}")
            print(f"  - Average probability (non-zero cells): {np.sum(grid)/nonzero_cells:.5f}")
        else:
            print("  - No valid spawn cells found!")


def plot_distribution(grid, goal_pos=None, lava_positions=None, title="Spawn Distribution", 
                      save_path=None, show=True):
    """
    Create a visual plot of the probability distribution.
    
    Parameters:
    ----------
    grid : numpy.ndarray
        Probability distribution array
    goal_pos : tuple (x, y) or None
        Position of the goal (if known)
    lava_positions : list of tuples (x, y) or None
        Positions of lava cells (if known)
    title : str
        Plot title
    save_path : str or None
        Path to save the visualization
    show : bool
        Whether to display the plot
    """
    height, width = grid.shape
    
    # Create a copy for plotting
    plot_grid = grid.copy()
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create heatmap
    im = ax.imshow(plot_grid, origin='upper', cmap='viridis')
    plt.colorbar(im, ax=ax, label='Probability')
    
    # Add grid lines
    ax.set_xticks(np.arange(-.5, width, 1), minor=True)
    ax.set_yticks(np.arange(-.5, height, 1), minor=True)
    ax.grid(which='minor', color='w', linestyle='-', linewidth=0.5, alpha=0.3)
    
    # Show coordinates
    ax.set_xticks(np.arange(0, width, 1))
    ax.set_yticks(np.arange(0, height, 1))
    
    # Add markers for goal and lava
    if goal_pos:
        ax.plot(goal_pos[0], goal_pos[1], 'r*', markersize=15, label='Goal')
    
    if lava_positions and len(lava_positions) > 0:
        lava_x, lava_y = zip(*lava_positions)
        ax.scatter(lava_x, lava_y, c='red', marker='x', s=100, label='Lava')
    
    # Annotate cells with probability values
    for y in range(height):
        for x in range(width):
            # Skip cells with zero probability
            if plot_grid[y, x] > 0:
                text = ax.text(x, y, f"{plot_grid[y, x]:.4f}",
                            ha="center", va="center", color="w", fontsize=8)
    
    # Set title and labels
    ax.set_title(title)
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    
    # Add legend if we have special markers
    if goal_pos or (lava_positions and len(lava_positions) > 0):
        ax.legend(loc='upper right')
    
    # Save plot if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    # Show or close
    if show:
        plt.show()
    else:
        plt.close()


def generate_ascii_visualization(grid, goal_pos=None, lava_positions=None, title=None):
    """
    Generate an ASCII visualization of the probability distribution.
    
    Parameters:
    ----------
    grid : numpy.ndarray
        Probability distribution array
    goal_pos : tuple (x, y) or None
        Position of the goal (if known)
    lava_positions : list of tuples (x, y) or None
        Positions of lava cells (if known)
    title : str or None
        Optional title to display
        
    Returns:
    -------
    str
        ASCII visualization string
    """
    if title:
        result = f"\n=== {title} ===\n"
    else:
        result = "\n"
    
    height, width = grid.shape
    
    if lava_positions is None:
        lava_positions = []
    
    # Map probability to symbols:
    # ' ' (0), '.' (>0-0.01), ':' (0.01-0.05), '•' (0.05-0.1), 'o' (0.1-0.2), 'O' (0.2-0.3), '@' (>0.3)
    symbols = [' ', '.', ':', '•', 'o', 'O', '@']
    thresholds = [0, 0.01, 0.05, 0.1, 0.2, 0.3]
    
    # Find maximum probability value for scaling
    max_prob = np.max(grid) if np.max(grid) > 0 else 1.0
    
    # Draw the grid with symbols
    for y in range(height):
        line = ""
        for x in range(width):
            if goal_pos and (x, y) == goal_pos:
                line += "G"  # Goal
            elif (x, y) in lava_positions:
                line += "L"  # Lava
            else:
                # Get normalized probability and map to symbol
                prob = grid[y, x]
                idx = 0
                for t in thresholds:
                    if prob > t:
                        idx += 1
                
                line += symbols[idx]
        
        result += line + "\n"
    
    # Add legend
    result += "\nLegend:\n"
    result += "G = Goal, L = Lava\n"
    result += f"Probability symbols: {' '.join(symbols)}\n"
    result += f"From 0 (space) to >0.3 (@)\n"
    
    return result


def visualize_wrapper_distribution(wrapper, title="Current Spawn Distribution", 
                                   save_path=None, show=True, print_values=True):
    """
    Visualize the distribution from a FlexibleSpawnWrapper.
    
    Parameters:
    ----------
    wrapper : FlexibleSpawnWrapper
        The wrapper instance to visualize
    title : str
        Plot title
    save_path : str or None
        Path to save the visualization
    show : bool
        Whether to display the plot
    print_values : bool
        Whether to also print the numeric values
    """
    if not hasattr(wrapper, 'current_distribution') or wrapper.current_distribution is None:
        print("Error: Wrapper has no current_distribution")
        return
    
    # Get distribution grid
    grid = wrapper.current_distribution.probabilities
    
    # Get environment information
    goal_pos = getattr(wrapper, 'goal_pos', None)
    
    # Find lava positions by scanning the grid
    lava_positions = []
    if hasattr(wrapper, 'unwrapped') and hasattr(wrapper.unwrapped, 'grid'):
        grid_obj = wrapper.unwrapped.grid
        for i in range(grid_obj.width):
            for j in range(grid_obj.height):
                cell = grid_obj.get(i, j)
                if cell and cell.type == 'lava':
                    lava_positions.append((i, j))
    
    # Print numeric distribution if requested
    if print_values:
        print_numeric_distribution(grid, goal_pos, lava_positions, title=title)
    
    # Plot the distribution
    plot_distribution(grid, goal_pos, lava_positions, title=title, 
                     save_path=save_path, show=show)


def inspect_env_for_spawn_wrapper(env):
    """
    Recursively search for FlexibleSpawnWrapper in environment.
    
    Parameters:
    ----------
    env : gym.Env
        Environment to inspect
    
    Returns:
    -------
    wrapper : FlexibleSpawnWrapper or None
        Found wrapper or None if not found
    """
    if isinstance(env, FlexibleSpawnWrapper):
        return env
    
    if hasattr(env, 'env'):
        return inspect_env_for_spawn_wrapper(env.env)
    
    return None
