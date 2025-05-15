from typing import List, Tuple, Optional, Callable, Any, Dict, Union
import numpy as np

class State:
    """
    Represents a state in an environment with position, orientation, type, and neighbors.
    
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
        2: [(0, 1), (-1, 1), (1, 1)],      # Down: forward=down, diag-left=down-left, diag-right=down-right
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
                        Access as env_tensor[x, y] to get the cell type at position (x, y).
        
        Returns:
            The assigned state type as a string
        """
        x, y, _ = self.state
        
        # Check if coordinates are within bounds
        if (0 <= x < env_tensor.shape[0] and 0 <= y < env_tensor.shape[1]):
            # Get the type from the tensor
            self.type = env_tensor[x, y]
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
            # Check if the coordinates are within the bounds of the environment tensor
            if (0 <= new_x < env_tensor.shape[0] and 
                0 <= new_y < env_tensor.shape[1]):
                
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
    def is_diagonal_safe(self, current_state, neighbor_state, orientation, move_type, env_tensor):
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
        
        # Check each cell for lava
        for cell_x, cell_y in cells_to_check:
            if (0 <= cell_x < env_tensor.shape[0] and 0 <= cell_y < env_tensor.shape[1]):
                if env_tensor[cell_x, cell_y] == "lava":
                    return False  # Not safe if any adjacent cell is lava
        
        return True  # Safe if no adjacent cells are lava
    
    def generate_valid_neighbors(self, env_tensor: np.ndarray, valid_types: Dict[str, List[str]] = None) -> Dict[str, List[Tuple[int, int, int]]]:
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
            x, y, orientation = neighbor_state
            
            # Skip if out of bounds
            if not (0 <= x < env_tensor.shape[0] and 0 <= y < env_tensor.shape[1]):
                continue
                
            # Get the type of this neighbor from the environment tensor
            neighbor_type = env_tensor[x, y]
            
            # DANGEROUS: Includes lava cells
            if neighbor_type in valid_types["dangerous"]:
                result["dangerous"].append(neighbor_state)
            
            # STANDARD: Only includes floor cells (no lava)
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
                    if self.is_diagonal_safe(self.state, neighbor_state, current_orientation, move_type, env_tensor):
                        result["conservative"].append(neighbor_state)
        
        # Store results in the object
        self.valid_dangerous = result["dangerous"]
        self.valid_standard = result["standard"]
        self.valid_conservative = result["conservative"]
        
        # For backward compatibility, set valid_neighbors to standard
        self.valid_neighbors = self.valid_standard
        
        return result
    
    def populate_object(self, env_tensor: np.ndarray, valid_types: Dict[str, List[str]] = None) -> None:
        """
        Populate all the object's information by running the necessary methods in sequence.
        
        Args:
            env_tensor: 3D numpy array representing the environment.
            valid_types: Dictionary of valid types for different neighbor categories.
        """
        # First, assign the state type
        self.assign_state_type(env_tensor)
        
        # Generate feasible neighbors
        self.generate_feasible_neighbors(env_tensor)
        
        # Generate valid neighbors in different categories
        self.generate_valid_neighbors(env_tensor, valid_types)
        
        # Additional population methods can be added here as needed


# Test function to demonstrate the behavior
def test_state_neighbors():
    """
    Test the State class with a simple environment to demonstrate neighbor generation
    """
    # Create a simple environment
    # 0 = floor, 1 = wall, 2 = lava, 3 = goal
    env_map = np.array([
        [0, 0, 0, 0, 0],
        [0, 0, 2, 0, 0],
        [0, 2, 0, 2, 0],
        [0, 0, 2, 0, 0],
        [0, 0, 0, 0, 3]
    ])
    
    # Map numeric codes to string labels
    type_map = {
        0: "floor",
        1: "wall",
        2: "lava",
        3: "goal"
    }
    
    # Convert numeric map to string labels
    env_tensor = np.empty(env_map.shape, dtype=object)
    for i in range(env_map.shape[0]):
        for j in range(env_map.shape[1]):
            env_tensor[i, j] = type_map[env_map[i, j]]
    
    # Print the environment
    print("Environment:")
    for row in env_tensor:
        print([cell.ljust(5) for cell in row])
    print()
    
    # Define valid types for each neighbor category
    valid_types = {
        "dangerous": ["floor", "goal", "lava"],
        "standard": ["floor", "goal"],
        "conservative": ["floor", "goal"]
    }
    
    # Test states in different positions and orientations
    test_states = [
        ((2, 2, 0), "Center with lava around"),  # Center, facing up
        ((0, 2, 1), "Edge case with lava nearby"),  # Top edge, facing right
        ((2, 0, 2), "Near corner case"),  # Left edge, facing down
        ((4, 4, 3), "Goal position")   # Bottom-right (goal), facing left
    ]
    
    # Test each state
    for state_coords, description in test_states:
        print(f"=== Testing {description}: {state_coords} ===")
        
        # Create a state object
        state = State(state_coords)
        
        # Populate the state
        state.populate_object(env_tensor, valid_types)
        
        # Print state info
        print(f"Position: ({state_coords[0]}, {state_coords[1]})")
        print(f"Orientation: {state_coords[2]} ({['up', 'right', 'down', 'left'][state_coords[2]]})")
        print(f"Type: {state.type}")
        
        # Print neighbors
        neighbor_types = {
            "feasible_neighbors": "Feasible",
            "valid_dangerous": "Dangerous",
            "valid_standard": "Standard",
            "valid_conservative": "Conservative"
        }
        
        for attr_name, display_name in neighbor_types.items():
            neighbors = getattr(state, attr_name)
            print(f"\n{display_name} neighbors:")
            
            if not neighbors:
                print("  None")
                continue
                
            for i, neighbor in enumerate(neighbors):
                x, y, o = neighbor
                move_type = state.identify_move_type(state.state, neighbor, state.state[2])
                print(f"  {i+1}. ({x}, {y}, {o}) - {['up', 'right', 'down', 'left'][o]} - {move_type}")
                
                # For conservative neighbors, explain why diagonals are safe
                if move_type.startswith("diagonal") and attr_name == "valid_conservative":
                    # Explain why this diagonal is safe
                    cx, cy, co = state.state
                    
                    # For diagonal-left, check forward and left
                    # For diagonal-right, check forward and right
                    forward = state.movement_vectors[co][0]
                    
                    if move_type == "diagonal-left":
                        left = state.left_vectors[co]
                        cell1 = (cx + forward[0], cy + forward[1])
                        cell2 = (cx + left[0], cy + left[1])
                        print(f"     Safe because adjacent cells are not lava:")
                        if (0 <= cell1[0] < env_tensor.shape[0] and 0 <= cell1[1] < env_tensor.shape[1]):
                            print(f"     - Forward ({cell1[0]}, {cell1[1]}): {env_tensor[cell1[0], cell1[1]]}")
                        else:
                            print(f"     - Forward ({cell1[0]}, {cell1[1]}): out of bounds")
                            
                        if (0 <= cell2[0] < env_tensor.shape[0] and 0 <= cell2[1] < env_tensor.shape[1]):
                            print(f"     - Left ({cell2[0]}, {cell2[1]}): {env_tensor[cell2[0], cell2[1]]}")
                        else:
                            print(f"     - Left ({cell2[0]}, {cell2[1]}): out of bounds")
                    else:
                        right = state.right_vectors[co]
                        cell1 = (cx + forward[0], cy + forward[1])
                        cell2 = (cx + right[0], cy + right[1])
                        print(f"     Safe because adjacent cells are not lava:")
                        if (0 <= cell1[0] < env_tensor.shape[0] and 0 <= cell1[1] < env_tensor.shape[1]):
                            print(f"     - Forward ({cell1[0]}, {cell1[1]}): {env_tensor[cell1[0], cell1[1]]}")
                        else:
                            print(f"     - Forward ({cell1[0]}, {cell1[1]}): out of bounds")
                            
                        if (0 <= cell2[0] < env_tensor.shape[0] and 0 <= cell2[1] < env_tensor.shape[1]):
                            print(f"     - Right ({cell2[0]}, {cell2[1]}): {env_tensor[cell2[0], cell2[1]]}")
                        else:
                            print(f"     - Right ({cell2[0]}, {cell2[1]}): out of bounds")
        
        print("\n" + "="*50 + "\n")


if __name__ == "__main__":
    test_state_neighbors() 