#!/usr/bin/env python3
"""
Example script to extract a DQN agent model and save its components.
This script simplifies the extraction process with a basic command-line interface.
"""

import os
import sys
import traceback

# Get the absolute path to the current file
current_file_path = os.path.abspath(__file__)
# Then, get the directory of the current file
current_dir = os.path.dirname(current_file_path)
# Get the parent directory (project root)
project_root = os.path.dirname(current_dir)
# Add project root to path
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    print(f"Added project root to Python path: {project_root}")

# Now import the extractor
try:
    from ModelAnalysis.ModelExtraction import DQNModelExtractor
    print("Successfully imported DQNModelExtractor")
except ImportError as e:
    print(f"Error importing DQNModelExtractor: {e}")
    print(f"Python path: {sys.path}")
    
    # Try alternative import paths
    try:
        print("Trying relative import...")
        from .ModelExtraction import DQNModelExtractor
        print("Successfully imported DQNModelExtractor using relative import")
    except ImportError:
        try:
            print("Trying direct import...")
            import sys
            sys.path.append(current_dir)  # Add current directory to path
            from ModelExtraction import DQNModelExtractor
            print("Successfully imported DQNModelExtractor from current directory")
        except ImportError as e2:
            print(f"All import attempts failed: {e2}")
            sys.exit(1)

def main():
    """
    Extract neural network model from a specified DQN agent.
    Usage: python extract_agent_model.py <agent_path>
    """
    if len(sys.argv) < 2:
        print("Usage: python extract_agent_model.py <agent_path> [output_dir]")
        print("Example: python extract_agent_model.py ../DebuggingAgents/LavaS11N5_5_exp_10m.zip")
        sys.exit(1)
    
    agent_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else None
    
    # Check if the path is absolute or relative
    if not os.path.isabs(agent_path):
        # If it's not in the current directory, try in the project root
        if not os.path.exists(agent_path):
            potential_path = os.path.join(project_root, agent_path)
            if os.path.exists(potential_path):
                agent_path = potential_path
                print(f"Using absolute path: {agent_path}")
    
    if not os.path.exists(agent_path):
        print(f"Error: Agent file not found: {agent_path}")
        print(f"Current directory: {os.getcwd()}")
        sys.exit(1)
    
    print(f"Extracting model from agent: {agent_path}")
    
    try:
        # Create the model extractor and extract the model
        extractor = DQNModelExtractor(
            agent_path=agent_path,
            output_dir=output_dir,
            verbose=True
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