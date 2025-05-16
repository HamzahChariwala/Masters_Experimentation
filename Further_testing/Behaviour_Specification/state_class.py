from typing import List, Tuple, Optional, Callable, Any, Dict, Union
import numpy as np

class State:
    """
    Represents a state in an environment with position, orientation, type, and neighbors.
    
    IMPORTANT NOTE ABOUT COORDINATE SYSTEM:
    - State tuples are defined as (x, y, orientation) where:
      * x: horizontal axis (increases to the right)
      * y: vertical axis (increases downward)
      * orientation: 0=up, 1=right, 2=down, 3=left
    - Environment tensor is accessed as tensor[y, x] because NumPy uses [row, column] indexing
    
    Attributes:
        state (Tuple[int, int, int]): Tuple of (x, y, orientation) representing the state
        type (str): Type of the state (e.g., "floor", "wall", "lava")
        feasible_neighbors (List[Tuple[int, int, int]]): List of feasible neighboring states as tuples
        valid_neighbors (List[Tuple[int, int, int]]): List of valid neighboring states after filtering
        valid_dangerous (List[Tuple[int, int, int]]): Valid neighbors with permissive criteria (more options)
        valid_standard (List[Tuple[int, int, int]]): Valid neighbors with standard balance criteria
        valid_conservative (List[Tuple[int, int, int]]): Valid neighbors with restrictive criteria (safer options)
    """
    
    # Class-level vectors - will be available to all methods
    # Movement vectors for each orientation - used to identify moves
    # Format: [forward, diag-left, diag-right] = [(dx, dy), (dx, dy), (dx, dy)]
    movement_vectors = {
        0: [(0, -1), (1, -1), (-1, -1)],   # Up: forward=up, diag-left=up-right, diag-right=up-left
        1: [(1, 0), (1, -1), (1, 1)],      # Right: forward=right, diag-left=up-right, diag-right=down-right
        2: [(0, 1), (1, 1), (-1, 1)],      # Down: forward=down, diag-left=down-right, diag-right=down-left
        3: [(-1, 0), (-1, -1), (-1, 1)]    # Left: forward=left, diag-left=up-left, diag-right=down-left
    }
    
    # Right and left vectors for each orientation
    right_vectors = {
        0: (1, 0),    # Up: right is (1, 0)
        1: (0, 1),    # Right: right is (0, 1)
        2: (-1, 0),   # Down: right is (-1, 0)
        3: (0, -1)    # Left: right is (0, -1)
    }
    
    left_vectors = {
        0: (-1, 0),   # Up: left is (-1, 0)
        1: (0, -1),   # Right: left is (0, -1)
        2: (1, 0),    # Down: left is (1, 0)
        3: (0, 1)     # Left: left is (0, 1)
    }
    
    def __init__(
        self, 
        state: Tuple[int, int, int], 
        state_type: str = "", 
        feasible_neighbors: Optional[List[Tuple[int, int, int]]] = None
    ):
        """
        Initialize a State object.
        
        Args:
            state (Tuple[int, int, int]): (x, y, orientation) coordinate tuple
            state_type (str, optional): Type of the state (e.g., "floor", "wall", "lava"). 
                                       If not provided, it will be assigned later.
            feasible_neighbors (List[Tuple[int, int, int]], optional): List of feasible neighboring states
        """
        self.state = state
        self.type = state_type
        self.feasible_neighbors = feasible_neighbors if feasible_neighbors is not None else []
        self.valid_neighbors = []
        self.valid_dangerous = []
        self.valid_standard = []
        self.valid_conservative = []
    
    def assign_state_type(self, env_tensor: np.ndarray) -> str:
        """
        Assign the state type based on the environment tensor.
        
        Args:
            env_tensor: Numpy array representing the environment.
                        Access as env_tensor[y, x] to get the cell type at position (x, y).
        
        Returns:
            The assigned state type as a string
        """
        x, y, _ = self.state
        
        # Check if coordinates are within bounds
        if (0 <= x < env_tensor.shape[1] and 0 <= y < env_tensor.shape[0]):
            # Get the type from the tensor
            # IMPORTANT: The indices need to be [y, x] because numpy uses [row, column]
            self.type = env_tensor[y, x]
        else:
            # If coordinates are out of bounds, mark as invalid
            self.type = "invalid"
            
        return self.type
    
    def generate_feasible_neighbors(self, env_tensor: np.ndarray, valid_types: List[str] = None) -> List[Tuple[int, int, int]]:
        """
        Generate feasible neighboring states for this state considering rotations and positional moves.
        
        Notes on coordinate system:
        - x: horizontal axis (increases to the right)
        - y: vertical axis (increases downward)
        - orientation: 0=up, 1=right, 2=down, 3=left
        
        Each state has two rotational neighbors and potentially three positional neighbors:
        1. Rotational: turn left and turn right (always valid)
        2. Positional: move forward, move diagonally left-forward, move diagonally right-forward
           (only pruned if coordinates would be outside the environment bounds)
        
        Args:
            env_tensor: 3D numpy array representing the environment.
                        Used only to determine the environment bounds.
            valid_types: Not used in this method, included for compatibility with the interface.
        
        Returns:
            List of feasible neighboring states as tuples (x, y, orientation)
        """
        neighbors = []
        x, y, orientation = self.state
        
        # Add rotational neighbors (always valid)
        # Left rotation: (orientation + 3) % 4
        # Right rotation: (orientation + 1) % 4
        neighbors.append((x, y, (orientation + 3) % 4))  # Turn left
        neighbors.append((x, y, (orientation + 1) % 4))  # Turn right
        
        # Get potential positional moves based on current orientation
        forward, diag_left, diag_right = self.movement_vectors[orientation]
        
        # Generate the three potential positional moves (forward, diag-left, diag-right)
        potential_moves = [
            (x + forward[0], y + forward[1], orientation),
            (x + diag_left[0], y + diag_left[1], orientation),
            (x + diag_right[0], y + diag_right[1], orientation)
        ]
        
        # Check each potential move for validity of coordinates only
        for new_x, new_y, new_orientation in potential_moves:
            # Check if coordinates are within bounds, using correct dimensions
            # env_tensor shape is (height, width) with indices [y, x]
            if (0 <= new_x < env_tensor.shape[1] and 
                0 <= new_y < env_tensor.shape[0]):
                
                # Add to neighbors if coordinates are valid, regardless of type
                neighbors.append((new_x, new_y, new_orientation))
        
        self.feasible_neighbors = neighbors
        return neighbors
    
    # Helper to check if a move is a rotation (same position, different orientation)
    def is_rotation(self, current_state, neighbor_state):
        cx, cy, _ = current_state
        nx, ny, _ = neighbor_state
        return cx == nx and cy == ny
    
    # Helper to identify move type (rotation, forward, diagonal-left, diagonal-right)
    def identify_move_type(self, current_state, neighbor_state, orientation):
        cx, cy, _ = current_state
        nx, ny, _ = neighbor_state
        
        # If same position, it's a rotation
        if cx == nx and cy == ny:
            return "rotation"
            
        # Get movement vectors for this orientation
        forward, diag_left, diag_right = self.movement_vectors[orientation]
        
        # Check which type of move this is
        if nx - cx == forward[0] and ny - cy == forward[1]:
            return "forward"
        elif nx - cx == diag_left[0] and ny - cy == diag_left[1]:
            return "diagonal-left"
        elif nx - cx == diag_right[0] and ny - cy == diag_right[1]:
            return "diagonal-right"
        else:
            return "unknown"
    
    # Helper to check if a diagonal move is safe for conservative agent
    def is_diagonal_safe(self, current_state, neighbor_state, orientation, move_type, env_tensor, debug=False):
        """
        Check if a diagonal move is safe for the conservative agent.
        
        Args:
            current_state: Current state tuple (x, y, orientation)
            neighbor_state: Neighbor state tuple to check (x, y, orientation)
            orientation: Current orientation (0-3)
            move_type: Type of move ("diagonal-left" or "diagonal-right")
            env_tensor: Environment tensor with cell types
            debug: Whether to print debug information (default: False)
            
        Returns:
            bool: True if the move is safe, False if unsafe (near lava)
        """
        cx, cy, _ = current_state
        
        # Get the vectors for the current orientation
        forward, diag_left, diag_right = self.movement_vectors[orientation]
        
        # Check the appropriate cells based on move type
        if move_type == "diagonal-left":
            # For diagonal-left, check forward and left cells
            cells_to_check = [
                (cx + forward[0], cy + forward[1]),  # Forward cell
                (cx + self.left_vectors[orientation][0], cy + self.left_vectors[orientation][1])  # Left cell
            ]
        elif move_type == "diagonal-right":
            # For diagonal-right, check forward and right cells
            cells_to_check = [
                (cx + forward[0], cy + forward[1]),  # Forward cell
                (cx + self.right_vectors[orientation][0], cy + self.right_vectors[orientation][1])  # Right cell
            ]
        else:
            # Not a diagonal move
            return True
        
        # Print debugging information if debug is enabled
        if debug:
            print(f"DIAGONAL SAFETY CHECK: {move_type} move from {current_state} to {neighbor_state}")
            print(f"  Checking cells: {cells_to_check}")
        
        # Check each cell for lava
        for cell_x, cell_y in cells_to_check:
            # Make sure we check within the environment bounds
            if (0 <= cell_x < env_tensor.shape[1] and 0 <= cell_y < env_tensor.shape[0]):
                # IMPORTANT: Environment tensor is indexed as [y, x]
                cell_type = env_tensor[cell_y, cell_x]
                
                if debug:
                    print(f"  Cell ({cell_x}, {cell_y}) has type: {cell_type}")
                
                if cell_type == "lava":
                    if debug:
                        print(f"  UNSAFE: Found lava at ({cell_x}, {cell_y})")
                    return False  # Not safe if any adjacent cell is lava
        
        if debug:
            print(f"  SAFE: No lava found in adjacent cells")
        return True  # Safe if no adjacent cells are lava
    
    def generate_valid_neighbors(self, env_tensor: np.ndarray, valid_types: Dict[str, List[str]] = None, debug: bool = False) -> Dict[str, List[Tuple[int, int, int]]]:
        """
        Generate different sets of valid neighbors from feasible neighbors using different criteria.
        
        Neighbor types:
        - Dangerous: Will include lava cells - the agent is willing to step into lava
        - Standard: Only includes floor cells - the agent avoids lava but takes otherwise valid moves
        - Conservative: Avoids diagonals where adjacent cells are lava - the agent is cautious
          about diagonal moves that might be risky (checks adjacent cells for safety)
        
        Args:
            env_tensor: 3D numpy array representing the environment.
            valid_types: Optional dictionary to override default valid types.
                        If not provided, defaults are used.
            debug: Whether to print debug information (default: False)
        
        Returns:
            Dictionary of neighbor lists for each category
        """
        # Default valid types if not provided
        if valid_types is None:
            valid_types = {
                "dangerous": ["floor", "goal", "lava"],  # Includes lava - agent will step into lava
                "standard": ["floor", "goal"],           # Standard safe moves - avoids lava
                "conservative": ["floor", "goal"]        # Same as standard, but with extra checks for diagonals
            }
        
        # Initialize results
        result = {
            "dangerous": [],
            "standard": [],
            "conservative": []
        }
        
        # Ensure we have feasible neighbors to work with
        if not self.feasible_neighbors:
            return result
        
        # Get current state coordinates and orientation
        current_x, current_y, current_orientation = self.state
        
        # Process each feasible neighbor
        for neighbor_state in self.feasible_neighbors:
            nx, ny, orientation = neighbor_state
            
            # Skip if out of bounds
            if not (0 <= nx < env_tensor.shape[1] and 0 <= ny < env_tensor.shape[0]):
                continue
                
            # Get the type of this neighbor from the environment tensor
            # IMPORTANT: Access as env_tensor[y, x] since numpy uses [row, column]
            neighbor_type = env_tensor[ny, nx]
            
            # DANGEROUS: Includes lava cells
            if neighbor_type in valid_types["dangerous"]:
                result["dangerous"].append(neighbor_state)
            
            # STANDARD: Only includes floor and goal cells (no lava)
            if neighbor_type in valid_types["standard"]:
                result["standard"].append(neighbor_state)
            
            # CONSERVATIVE: Like standard but with extra checks for diagonals
            if neighbor_type in valid_types["conservative"]:
                # Identify the type of move
                move_type = self.identify_move_type(self.state, neighbor_state, current_orientation)
                
                # If it's a rotation or forward move, add it directly
                if move_type in ["rotation", "forward"]:
                    result["conservative"].append(neighbor_state)
                # For diagonal moves, check if they're safe
                elif move_type in ["diagonal-left", "diagonal-right"]:
                    if self.is_diagonal_safe(self.state, neighbor_state, current_orientation, move_type, env_tensor, debug):
                        result["conservative"].append(neighbor_state)
        
        # Store results in the object
        self.valid_dangerous = result["dangerous"]
        self.valid_standard = result["standard"]
        self.valid_conservative = result["conservative"]
        
        # For backward compatibility, set valid_neighbors to standard
        self.valid_neighbors = self.valid_standard
        
        return result
    
    def populate_object(self, env_tensor: np.ndarray, valid_types: Dict[str, List[str]] = None, debug: bool = False) -> None:
        """
        Populate all the object's information by running the necessary methods in sequence.
        
        Args:
            env_tensor: 3D numpy array representing the environment.
            valid_types: Dictionary of valid types for different neighbor categories.
            debug: Whether to print debug information (default: False)
        """
        # First, assign the state type
        self.assign_state_type(env_tensor)
        
        # Generate feasible neighbors
        self.generate_feasible_neighbors(env_tensor)
        
        # Generate valid neighbors in different categories
        self.generate_valid_neighbors(env_tensor, valid_types, debug)
        
        # Additional population methods can be added here as needed
