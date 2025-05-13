#!/usr/bin/env python3
"""
Script to extract DQN agent models and save components for analysis.
This script is designed to work with the Agent_Storage directory structure.
"""

import os
import sys
import traceback
import argparse

# Get the absolute path to the current file
current_file_path = os.path.abspath(__file__)
# Then, get the directory of the current file
current_dir = os.path.dirname(current_file_path)
# Get the parent directory (Inference_Estimation)
inference_dir = os.path.dirname(current_dir)
# Get the project root (one level up)
project_root = os.path.dirname(inference_dir)

# Add necessary paths to sys.path
import_paths = [
    project_root,                                         # Project root
    inference_dir,                                        # Inference_Estimation directory
    os.path.join(project_root, "DRL_Training"),          # DRL_Training folder
    current_dir,                                          # ModelAnalysis directory
    os.path.join(project_root, "Agent_Storage")          # Agent_Storage directory
]

# Add all paths to sys.path
for path in import_paths:
    if os.path.exists(path) and path not in sys.path:
        sys.path.insert(0, path)
        print(f"Added to Python path: {path}")

# Now import the extractor
try:
    from ModelAnalysis.ModelExtraction import DQNModelExtractor
    print("Successfully imported DQNModelExtractor")
except ImportError as e:
    print(f"Error importing DQNModelExtractor: {e}")
    
    # Try alternative import paths
    try:
        from ModelExtraction import DQNModelExtractor
        print("Successfully imported DQNModelExtractor from current directory")
    except ImportError as e2:
        print(f"All import attempts failed: {e2}")
        sys.exit(1)

def find_agent_file(agent_name: str) -> str:
    """
    Find an agent file using the Agent_Storage directory structure.
    
    Args:
        agent_name: Name of the agent or path to agent file
        
    Returns:
        Path to the agent file
    """
    # Check if the input is already a complete path to a zip file
    if os.path.exists(agent_name) and agent_name.endswith('.zip'):
        return agent_name
        
    # Check if it's an agent name from Agent_Storage
    agent_storage_path = os.path.join(project_root, "Agent_Storage")
    
    # Try several possible locations/formats
    possible_paths = [
        agent_name,  # As is (might be relative or absolute path)
        os.path.join(agent_storage_path, agent_name, "agent", "agent.zip"),  # Standard Agent_Storage path
        os.path.join(agent_storage_path, agent_name, f"{agent_name}.zip"),  # Alternative format
        os.path.join(agent_storage_path, agent_name, "agent.zip"),  # Simple format
        os.path.join(project_root, agent_name),  # Project root
        os.path.join(project_root, "DRL_Training", agent_name),  # Legacy path in DRL_Training
        os.path.join(project_root, "DRL_Training", f"{agent_name}.zip"),  # Legacy path with .zip
        os.path.join(project_root, "DRL_Training/DebuggingAgents", agent_name), # Legacy DebuggingAgents path
        os.path.join(project_root, "DRL_Training/DebuggingAgents", f"{agent_name}.zip") # Legacy with .zip
    ]
    
    # Find the first path that exists
    for path in possible_paths:
        if os.path.exists(path):
            print(f"Found agent at: {path}")
            return path
    
    # If no valid path found, return the standard expected path (for error message)
    return os.path.join(agent_storage_path, agent_name, "agent", "agent.zip")

def get_output_path(agent_path: str, output_dir: str = None) -> str:
    """
    Determine the output path for extracted model components.
    
    Args:
        agent_path: Path to the agent file
        output_dir: Optional custom output directory
        
    Returns:
        Path to use for output
    """
    if output_dir is not None:
        return output_dir
    
    # Parse the agent path to determine the agent name and structure
    agent_path = os.path.abspath(agent_path)
    agent_dir = os.path.dirname(agent_path)
    
    # Check if we're in the Agent_Storage structure
    agent_storage_path = os.path.join(project_root, "Agent_Storage")
    
    if agent_storage_path in agent_path:
        # We're in Agent_Storage structure
        # Extract the agent name from the path
        parts = agent_path.split(agent_storage_path)[1].split(os.sep)
        if len(parts) >= 2:
            agent_name = parts[1]  # First non-empty part after Agent_Storage
            return os.path.join(agent_storage_path, agent_name, "extracted_model")
    
    # If we're not in Agent_Storage or can't determine the agent name,
    # just use the base filename without extension
    agent_basename = os.path.basename(agent_path)
    if agent_basename.endswith('.zip'):
        agent_basename = agent_basename[:-4]
        
    return os.path.join(agent_storage_path, agent_basename, "extracted_model")

def main():
    """
    Extract neural network model from a specified DQN agent.
    """
    parser = argparse.ArgumentParser(description='Extract neural network models from DRL agents')
    parser.add_argument('--path', type=str, required=True,
                        help='Path to the agent folder in Agent_Storage or path to the agent file')
    parser.add_argument('output_dir', type=str, nargs='?', default=None, 
                        help='Directory to save extracted models (defaults to Agent_Storage/<agent_name>/extracted_model)')
    parser.add_argument('--verbose', '-v', action='count', default=1,
                        help='Increase verbosity (can be used multiple times)')
    
    args = parser.parse_args()
    
    # Require path parameter
    if not args.path:
        print("Error: No agent path specified. Please use --path parameter.")
        parser.print_help()
        sys.exit(1)
    
    # Find the agent file
    agent_path = find_agent_file(args.path)
    
    # Determine the agent name from the path
    agent_name = os.path.basename(agent_path)
    if agent_name.endswith('.zip'):
        agent_name = agent_name[:-4]  # Remove .zip extension
    
    # Determine output directory
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = get_output_path(agent_path)
    
    if not os.path.exists(agent_path):
        print(f"Error: Agent file not found: {args.path}")
        print(f"Expected location: {agent_path}")
        print(f"Please ensure the agent exists in the Agent_Storage directory structure.")
        sys.exit(1)
    
    print(f"Extracting model from agent: {agent_path}")
    print(f"Output will be saved to: {output_dir}")
    
    try:
        # Create the model extractor and extract the model
        extractor = DQNModelExtractor(
            agent_path=agent_path,
            output_dir=output_dir,
            verbose=args.verbose
        )
        
        # Save all model components
        saved_paths = extractor.save_model_components()
        
        print("\nExtraction complete!")
        print(f"Extracted files are saved to: {os.path.dirname(saved_paths['architecture'])}")
        print("\nYou can now use these extracted models for:")
        print("1. Analyzing the model architecture")
        print("2. Visualizing the neural network")
        print("3. Editing weights and model components") 
        print("4. Running inference without SB3 dependencies")
    except Exception as e:
        print(f"Error during extraction: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main() 