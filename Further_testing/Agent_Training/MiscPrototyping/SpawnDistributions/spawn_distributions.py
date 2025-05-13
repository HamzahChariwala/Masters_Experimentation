import numpy as np
import gymnasium as gym
from gymnasium import spaces
import random
import matplotlib.pyplot as plt
import os


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
        zero_mask = (self.probabilities == 0)
        
        # Linear interpolation
        self.probabilities = (1 - progress) * self.probabilities + progress * target_distribution.probabilities
        
        # Re-apply the zero mask to ensure these cells remain zero
        self.probabilities[zero_mask] = 0
        
        # Ensure the target's zero cells are also preserved when fully transitioning
        if progress > 0.99:  # Close to full transition
            target_zero_mask = (target_distribution.probabilities == 0)
            self.probabilities[target_zero_mask] = 0
        
        # Ensure valid distribution
        if np.sum(self.probabilities) > 0:
            self.probabilities /= np.sum(self.probabilities)
        else:
            # If all probabilities become zero, return to uniform over non-zero cells
            nonzero_mask = ~zero_mask
            if np.any(nonzero_mask):
                self.probabilities[nonzero_mask] = 1.0
                self.probabilities /= np.sum(self.probabilities)
        
        return self
    
    def build_sampling_map(self):
        """
        Build a cumulative probability mapping for sampling.
        
        This creates a flattened array that can be used with binary search to sample
        cells according to their probabilities.
        """
        # Flatten the probability array
        flat_probs = self.probabilities.flatten()
        
        # Create the cumulative sum
        self.cumulative_mapping = np.cumsum(flat_probs)
        
        # Ensure the final value is exactly 1.0 to avoid floating point issues
        self.cumulative_mapping[-1] = 1.0
        
        return self
    
    def sample(self):
        """
        Sample a position from the probability distribution.
        
        Returns:
        -------
        tuple (x, y) representing the sampled position
        """
        if self.cumulative_mapping is None:
            self.build_sampling_map()
        
        # Generate a random value between 0 and 1
        r = random.random()
        
        # Use binary search to find the index
        idx = np.searchsorted(self.cumulative_mapping, r)
        
        # Convert the flat index back to 2D coordinates
        y, x = divmod(idx, self.width)
        
        return x, y
    
    def plot(self, title="Probability Distribution", show=True, save_path=None):
        """
        Visualize the probability distribution as a heatmap.
        
        Parameters:
        ----------
        title : str
            Title for the plot
        show : bool
            Whether to display the plot
        save_path : str
            Path to save the plot image, or None to not save
        """
        plt.figure(figsize=(10, 8))
        plt.imshow(self.probabilities, cmap='viridis', origin='upper')
        plt.colorbar(label='Probability')
        plt.title(title)
        plt.xlabel('X')
        plt.ylabel('Y')
        
        # Add grid lines
        plt.grid(color='white', linestyle='-', linewidth=0.5, alpha=0.3)
        
        # Add coordinate labels
        plt.xticks(np.arange(self.width))
        plt.yticks(np.arange(self.height))
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        if show:
            plt.show()
        else:
            plt.close()
        
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
                    },
                    ...
                ]
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
        
        # Initialize timestep counter
        self.timestep = 0
        
        # Stage-based tracking
        self.current_stage = 0
        self.next_stage_transition = 0
        if stage_based_training and total_timesteps:
            if 'num_stages' in stage_based_training and 'distributions' in stage_based_training:
                self.stage_timesteps = total_timesteps / stage_based_training['num_stages']
                self.next_stage_transition = self.stage_timesteps
                
        # Placeholder for distribution maps
        self.current_distribution = None
        self.target_distribution = None
        self.valid_cells_mask = None
        
        # For tracking and visualization
        self.distribution_history = []
    
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
        
        # Store initial distribution for visualization
        initial_dist = self.current_distribution.probabilities.copy()
        self.distribution_history = [(0, initial_dist)]
    
    def _apply_distribution(self, dist_type, dist_params):
        """Apply a specific distribution type with given parameters."""
        params = dist_params or {}
        
        if dist_type == "uniform":
            self.current_distribution.uniform_distribution()
        
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
            
            # Create distribution centered on goal
            self.current_distribution.gaussian_from_point(
                self.goal_pos[0], self.goal_pos[1], sigma
            )
            
            # Invert if we want to favor positions far from goal
            if not favor_near:
                self.current_distribution.invert()
        
        elif dist_type == "distance_goal" and self.goal_pos:
            favor_near = params.get("favor_near", True)
            power = params.get("power", 2)
            
            # Create distribution centered on goal
            self.current_distribution.distance_based_from_point(
                self.goal_pos[0], self.goal_pos[1], favor_near, power
            )
        
        elif dist_type == "poisson_lava" or dist_type == "gaussian_lava":
            # Find all lava cells
            lava_positions = []
            for i in range(self.current_distribution.width):
                for j in range(self.current_distribution.height):
                    cell = self.unwrapped.grid.get(i, j)
                    if cell and cell.type == 'lava':
                        lava_positions.append((i, j))
            
            if lava_positions:
                # Set parameters based on distribution type
                if dist_type == "poisson_lava":
                    distribution_type = "poisson"
                    lambda_param = params.get("lambda_param", 2.0)
                    distribution_params = {"lambda_param": lambda_param}
                else:  # gaussian_lava
                    distribution_type = "gaussian"
                    sigma = params.get("sigma", 1.0)
                    distribution_params = {"sigma": sigma}
                
                # Create multi-point distribution
                favor_near = params.get("favor_near", True)
                self.current_distribution.multi_point_distribution(
                    lava_positions, distribution_type, distribution_params
                )
                
                # Invert if we want to favor positions far from lava
                if not favor_near:
                    self.current_distribution.invert()
            else:
                # Fall back to uniform if no lava found
                self.current_distribution.uniform_distribution()
        
        else:
            # Default to uniform distribution
            self.current_distribution.uniform_distribution()
    
    def step(self, action):
        """Execute a step and update the temporal distribution if enabled."""
        # Perform the environment step
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Update timestep counter
        self.timestep += 1
        
        # Check if we need to update the distribution
        distribution_updated = False
        
        # Update the distribution for stage-based training
        if (self.stage_based_training and self.total_timesteps and 
            'distributions' in self.stage_based_training and 
            len(self.stage_based_training['distributions']) > 1):
            
            # Check if we need to move to the next stage
            if self.timestep >= self.next_stage_transition:
                self.current_stage += 1
                num_stages = self.stage_based_training['num_stages']
                
                # Ensure we don't exceed the number of stages
                if self.current_stage < num_stages:
                    # Update next transition point
                    self.next_stage_transition = (self.current_stage + 1) * self.stage_timesteps
                    
                    # Get the next stage distribution
                    stage_dist = self.stage_based_training['distributions'][self.current_stage]
                    
                    # Apply new distribution
                    self._apply_distribution(stage_dist['type'], stage_dist.get('params', {}))
                    self.current_distribution.mask_cells(self.valid_cells_mask)
                    self.current_distribution.build_sampling_map()
                    
                    # Record the distribution change
                    self.distribution_history.append((self.timestep, self.current_distribution.probabilities.copy()))
                    
                    distribution_updated = True
        
        # Update the distribution if temporal transition is enabled (continuous transition)
        if not distribution_updated and self.temporal_transition and self.target_distribution and self.total_timesteps:
            # Calculate progress based on timestep
            rate = self.temporal_transition.get("rate", 1.0)
            progress = min(1.0, (self.timestep / self.total_timesteps) ** rate)
            
            # Interpolate between initial and target distributions
            self.current_distribution.temporal_interpolation(self.target_distribution, progress)
            
            # Re-apply valid cells mask to ensure goal and lava remain zero probability
            self.current_distribution.mask_cells(self.valid_cells_mask)
            
            # Rebuild the sampling map
            self.current_distribution.build_sampling_map()
            
            # Record the distribution change periodically (to avoid too many records)
            if self.timestep % (self.total_timesteps // 10) == 0:
                self.distribution_history.append((self.timestep, self.current_distribution.probabilities.copy()))
        
        return obs, reward, terminated, truncated, info
    
    def get_current_distribution_probabilities(self):
        """
        Returns the current distribution probabilities for external access.
        This method is used by visualization tools to directly access the distribution.
        
        Returns:
        -------
        numpy.ndarray
            The current probability distribution for spawn positions
        """
        if hasattr(self, 'current_distribution') and self.current_distribution is not None:
            return self.current_distribution.probabilities
        return None
    
    def visualize_distribution(self, title=None, save_path=None):
        """Visualize the current spawn distribution."""
        if self.current_distribution:
            if title is None:
                title = f"Spawn Distribution ({self.distribution_type})"
            
            self.current_distribution.plot(title=title, save_path=save_path)

    def reset_timestep(self):
        """Reset the timestep counter and stage tracking."""
        self.timestep = 0
        self.current_stage = 0
        self.distribution_history = [(0, self.current_distribution.probabilities.copy())]
        if self.stage_based_training and 'num_stages' in self.stage_based_training:
            self.stage_timesteps = self.total_timesteps / self.stage_based_training['num_stages']
            self.next_stage_transition = self.stage_timesteps
    
    def visualize_training_stages(self, output_dir=None, base_filename="stage_distribution"):
        """Visualize all stage distributions from training history."""
        if not self.distribution_history:
            print("No distribution history available.")
            return
        
        # Create output directory if provided
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        width, height = self.current_distribution.width, self.current_distribution.height
        
        # Plot each stage distribution
        for i, (timestep, dist) in enumerate(self.distribution_history):
            # Create a temporary distribution map for visualization
            vis_map = DistributionMap(width, height)
            vis_map.from_existing_distribution(dist)
            
            # Create title with stage and timestep info
            progress = self.total_timesteps and f" - {timestep / self.total_timesteps:.2%}" or ""
            stage_title = f"Distribution at timestep {timestep}{progress}"
            
            # Save path
            save_path = None
            if output_dir:
                save_path = os.path.join(output_dir, f"{base_filename}_{i:03d}.png")
            
            # Visualize
            vis_map.plot(title=stage_title, show=(save_path is None), save_path=save_path)
            
            if save_path:
                print(f"Saved stage distribution to {save_path}")
        
        # Create an animated visualization if we have multiple stages
        if len(self.distribution_history) > 1 and output_dir:
            self._create_training_animation(output_dir, base_filename)
            
    def validate_distributions(self):
        """
        Validate that the goal and lava positions have zero probability.
        
        Returns:
        -------
        dict: A validation report with information about:
            - Whether goal has zero probability
            - Whether all lava cells have zero probability
            - Any cells with non-zero probability that should be zero
        """
        # Get grid properties
        grid = self.unwrapped.grid
        width, height = grid.width, grid.height
        
        # Initialize validation report
        report = {
            "goal_zero_probability": True,
            "lava_zero_probability": True,
            "invalid_spawn_positions": []
        }
        
        # Check if the goal position has zero probability
        if self.goal_pos:
            goal_x, goal_y = self.goal_pos
            goal_prob = self.current_distribution.probabilities[goal_y, goal_x]
            if goal_prob > 0:
                report["goal_zero_probability"] = False
                report["invalid_spawn_positions"].append({"type": "goal", "position": self.goal_pos, "probability": goal_prob})
        
        # Check if lava positions have zero probability
        lava_positions = []
        for i in range(width):
            for j in range(height):
                cell = grid.get(i, j)
                if cell and cell.type == 'lava':
                    lava_positions.append((i, j))
                    lava_prob = self.current_distribution.probabilities[j, i]
                    if lava_prob > 0:
                        report["lava_zero_probability"] = False
                        report["invalid_spawn_positions"].append({"type": "lava", "position": (i, j), "probability": lava_prob})
        
        # Check for any other cells that should have zero probability but don't
        for i in range(width):
            for j in range(height):
                if self.valid_cells_mask[j, i] == 0:  # Should be zero probability
                    cell_prob = self.current_distribution.probabilities[j, i]
                    if cell_prob > 0:
                        cell = grid.get(i, j)
                        cell_type = cell.type if cell else "unknown"
                        report["invalid_spawn_positions"].append({"type": cell_type, "position": (i, j), "probability": cell_prob})
        
        # Validate target distribution if it exists
        if self.target_distribution:
            # Check if the goal position has zero probability in target
            if self.goal_pos:
                goal_x, goal_y = self.goal_pos
                goal_prob = self.target_distribution.probabilities[goal_y, goal_x]
                if goal_prob > 0:
                    report.setdefault("target_distribution_issues", []).append(
                        {"type": "goal", "position": self.goal_pos, "probability": goal_prob}
                    )
            
            # Check if lava positions have zero probability in target
            for i, j in lava_positions:
                lava_prob = self.target_distribution.probabilities[j, i]
                if lava_prob > 0:
                    report.setdefault("target_distribution_issues", []).append(
                        {"type": "lava", "position": (i, j), "probability": lava_prob}
                    )
        
        # Final validation result
        report["is_valid"] = (
            report["goal_zero_probability"] and 
            report["lava_zero_probability"] and 
            len(report["invalid_spawn_positions"]) == 0 and
            not report.get("target_distribution_issues", [])
        )
        
        return report

    def _create_training_animation(self, output_dir, base_filename):
        """Create an animated visualization of the training progression."""
        try:
            import matplotlib.animation as animation
            
            # Setup the figure
            fig, ax = plt.subplots(figsize=(10, 8))
            
            width, height = self.current_distribution.width, self.current_distribution.height
            
            # Get all distributions
            distributions = [dist for _, dist in self.distribution_history]
            timestamps = [ts for ts, _ in self.distribution_history]
            
            # Create plot
            im = ax.imshow(distributions[0], cmap='viridis', origin='upper', animated=True)
            plt.colorbar(im, ax=ax, label='Probability')
            ax.grid(color='white', linestyle='-', linewidth=0.5, alpha=0.3)
            
            # Add title with timestep info
            title = ax.set_title(f"Distribution at timestep {timestamps[0]}")
            
            def update(frame):
                """Update function for animation."""
                im.set_array(distributions[frame])
                progress = self.total_timesteps and f" - {timestamps[frame] / self.total_timesteps:.2%}" or ""
                title.set_text(f"Distribution at timestep {timestamps[frame]}{progress}")
                return [im, title]
            
            # Create the animation
            ani = animation.FuncAnimation(
                fig, update, frames=len(distributions), 
                interval=500, blit=False
            )
            
            # Save the animation
            animation_path = os.path.join(output_dir, f"{base_filename}_animation.gif")
            ani.save(animation_path, writer='pillow', fps=2)
            
            print(f"Saved training progression animation to {animation_path}")
            
            # Close the figure
            plt.close(fig)
            
        except ImportError as e:
            print(f"Could not create animation: {e}")
            print("Make sure matplotlib and Pillow are installed for animation support.") 