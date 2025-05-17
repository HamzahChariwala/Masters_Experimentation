import os
import json
import numpy as np

class AgentLogger:
    def export_to_json(self, output_path: str = None) -> str:
        """
        Export the agent behavior data to a JSON file.
        Format to match Dijkstra logs structure.
        
        Args:
            output_path (str, optional): Path to save the JSON file. 
                If None, saved to Agent_Evaluation/AgentLogs directory.
            
        Returns:
            str: Path to the saved JSON file
        """
        # Helper function to recursively convert NumPy arrays to lists
        def convert_numpy_to_list(item):
            if isinstance(item, np.ndarray):
                return item.tolist()
            elif isinstance(item, dict):
                return {k: convert_numpy_to_list(v) for k, v in item.items()}
            elif isinstance(item, (list, tuple)):
                return [convert_numpy_to_list(i) for i in item]
            else:
                return item
        
        # Build the output data structure to match Dijkstra log format
        # Format: {"environment": {"layout": [...], "1,1,0": {...}, ...}}
        
        # Get the environment tensor data
        env_layout = convert_numpy_to_list(self.env_tensor)
        
        # Setup the output data structure
        output_data = {
            "environment": {
                # Add the environment layout to the output
                "layout": env_layout,
            }
        }
        
        # Add all state data to the output
        for state_key, state_data in self.all_states_data.items():
            output_data["environment"][state_key] = convert_numpy_to_list(state_data)
        
        # Create the output path if not provided
        if output_path is None:
            output_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "AgentLogs"
            )
            # Create the directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)
            
            # Create the output file path
            output_path = os.path.join(
                output_dir,
                f"{self.env_id}-{self.seed}-agent.json"
            )
        
        # Save the data to a JSON file
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        return output_path 