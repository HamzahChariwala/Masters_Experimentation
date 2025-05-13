import os
import torch
import numpy as np
from stable_baselines3 import DQN
import argparse
import json
from typing import Dict, Any, Tuple, List, Optional
import sys
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path so we can import from it
# First, get the absolute path to the current file
current_file_path = os.path.abspath(__file__)
# Then, get the directory of the current file
current_dir = os.path.dirname(current_file_path)
# Get the parent directory (project root)
project_root = os.path.dirname(current_dir)
# Add project root to path
sys.path.insert(0, project_root)

try:
    # Now we should be able to import correctly
    from DRL_Training.EnvironmentEdits.BespokeEdits.FeatureExtractor import CustomCombinedExtractor
    print("Successfully imported CustomCombinedExtractor")
    FEATURE_EXTRACTOR_IMPORTED = True
except ImportError as e:
    print(f"Warning: Could not import CustomCombinedExtractor: {e}")
    print(f"Python path: {sys.path}")
    FEATURE_EXTRACTOR_IMPORTED = False


# Custom JSON encoder to handle NumPy types
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)


class DQNModelExtractor:
    """
    Extracts neural network models from DQN agents and saves them in a format
    that's convenient for loading, inference, and weight editing.
    """
    def __init__(self, agent_path: str, output_dir: str = None, verbose: bool = True):
        """
        Initialize the model extractor.
        
        Args:
            agent_path: Path to the saved DQN agent (zip file)
            output_dir: Directory to save extracted models
            verbose: Whether to print detailed information
        """
        self.agent_path = agent_path
        self.verbose = verbose
        
        # Default output directory is the same as the script location
        if output_dir is None:
            self.output_dir = os.path.dirname(os.path.abspath(__file__))
        else:
            self.output_dir = output_dir
            
        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Load the agent
        if self.verbose:
            print(f"Loading agent from {self.agent_path}")
        try:
            # Add project root to path again before loading the agent
            # This ensures that the pickle loader can find the required modules
            if project_root not in sys.path:
                sys.path.insert(0, project_root)
                
            self.agent = DQN.load(self.agent_path)
            print("Agent loaded successfully")
            
            # Detailed diagnostic information about policy and features extractor
            if self.verbose:
                print(f"Policy type: {type(self.agent.policy)}")
                print(f"Policy attributes: {dir(self.agent.policy)}")
                
            # Check if feature extractor exists and is properly loaded
            if hasattr(self.agent.policy, 'features_extractor'):
                features_extractor = self.agent.policy.features_extractor
                if features_extractor is not None:
                    print(f"Feature extractor found: {type(features_extractor)}")
                    if hasattr(features_extractor, '_features_dim'):
                        print(f"Feature dimension: {features_extractor._features_dim}")
                    elif hasattr(features_extractor, 'features_dim'):
                        print(f"Feature dimension: {features_extractor.features_dim}")
                else:
                    # Try to diagnose why it's None
                    print("Warning: Feature extractor is None")
                    print(f"Agent policy type: {type(self.agent.policy)}")
                    if hasattr(self.agent.policy, 'observation_space'):
                        print(f"Observation space: {self.agent.policy.observation_space}")
            else:
                print("Warning: Policy does not have a 'features_extractor' attribute")
                
        except Exception as e:
            print(f"Error loading agent: {e}")
            raise
        
        # Extract agent name from path
        self.agent_name = os.path.basename(self.agent_path).split('.')[0]
        
        # Check if feature extractor exists
        if self.agent.policy.features_extractor is None:
            print("Warning: Agent does not have a features extractor. Some functions will be limited.")
            print("This could be because:")
            print("1. The agent was trained without a custom feature extractor")
            print("2. The feature extractor class cannot be found in the current environment")
            print("3. There's a mismatch between the saved model structure and what's expected")
            
            # Try to reconstruct the extractor if needed
            if FEATURE_EXTRACTOR_IMPORTED:
                print("Attempting to reconstruct feature extractor from model architecture...")
                try:
                    obs_space = self.agent.observation_space
                    # Check if we can infer feature extractor details from the policy
                    if hasattr(self.agent.policy, 'q_net') and hasattr(self.agent.policy.q_net, 'features_extractor'):
                        print("Found features_extractor reference in q_net")
                except Exception as e:
                    print(f"Could not reconstruct feature extractor: {e}")
        
    def get_policy_network(self) -> torch.nn.Module:
        """
        Extract the policy network from the agent.
        
        Returns:
            The policy network as a PyTorch module
        """
        # For DQN, the policy is already a Q-network
        policy_net = self.agent.policy
        if self.verbose:
            print("Policy network architecture:")
            for name, param in policy_net.state_dict().items():
                print(f"  {name}: {param.shape}")
        return policy_net
    
    def get_q_network(self) -> torch.nn.Module:
        """
        Extract the Q-network from the agent.
        
        Returns:
            The Q-network as a PyTorch module
        """
        # For DQN, q_net is the action value network
        q_net = self.agent.q_net
        if self.verbose:
            print("Q-network architecture:")
            for name, param in q_net.state_dict().items():
                print(f"  {name}: {param.shape}")
        return q_net
    
    def get_feature_extractor(self) -> Optional[torch.nn.Module]:
        """
        Extract the feature extractor from the agent.
        
        Returns:
            The feature extractor as a PyTorch module, or None if not available
        """
        # The feature extractor is part of the policy
        feature_extractor = self.agent.policy.features_extractor
        if feature_extractor is None:
            if self.verbose:
                print("No feature extractor found in the agent.")
            return None
            
        if self.verbose:
            print("Feature extractor architecture:")
            for name, param in feature_extractor.state_dict().items():
                print(f"  {name}: {param.shape}")
        return feature_extractor
    
    def get_action_net(self) -> Optional[torch.nn.Module]:
        """
        Extract the action network (final layers after feature extraction).
        
        Returns:
            The action network as a PyTorch module, or None if not available
        """
        # For DQN, the action network is the q_net head after feature extraction
        try:
            action_net = self.agent.q_net.q_net
            if self.verbose:
                print("Action network architecture:")
                for name, param in action_net.state_dict().items():
                    print(f"  {name}: {param.shape}")
            return action_net
        except AttributeError:
            if self.verbose:
                print("Could not access action network directly. It might have a different structure.")
            return None
    
    def analyze_weight_statistics(self) -> Dict[str, Dict[str, float]]:
        """
        Analyze the statistics of weights and biases for each layer.
        
        Returns:
            Dictionary containing weight statistics for each layer.
        """
        stats = {}
        
        # Get Q-network state dict
        state_dict = self.agent.q_net.state_dict()
        
        for name, param in state_dict.items():
            # Skip target network parameters
            if 'target' in name:
                continue
                
            param_data = param.detach().cpu().numpy()
            
            # Calculate statistics
            layer_stats = {
                'min': float(np.min(param_data)),
                'max': float(np.max(param_data)),
                'mean': float(np.mean(param_data)),
                'std': float(np.std(param_data)),
                'median': float(np.median(param_data)),
                'sparsity': float((param_data == 0).sum() / param_data.size),
                'shape': list(param.shape),
                'num_params': int(np.prod(param.shape))
            }
            
            # Add abs stats
            layer_stats['abs_mean'] = float(np.mean(np.abs(param_data)))
            layer_stats['abs_max'] = float(np.max(np.abs(param_data)))
            
            # Add histogram data (10 bins)
            hist, bin_edges = np.histogram(param_data, bins=10)
            layer_stats['histogram'] = {
                'counts': hist.tolist(),
                'bin_edges': bin_edges.tolist()
            }
            
            stats[name] = layer_stats
            
        return stats
    
    def generate_network_summary(self) -> Dict[str, Any]:
        """
        Generate a comprehensive summary of the neural network including 
        layer sizes, parameter counts, and overall structure.
        
        Returns:
            Dictionary containing network summary information.
        """
        summary = {
            'total_params': 0,
            'trainable_params': 0,
            'non_trainable_params': 0,
            'layers': [],
            'observation_space': str(self.agent.observation_space),
            'action_space': str(self.agent.action_space),
            'model_type': 'DQN'
        }
        
        # Analyze policy layers, but exclude target network parameters
        for name, param in self.agent.policy.named_parameters():
            # Skip target network parameters (they're just copies used for training stability)
            if 'target' in name:
                continue
                
            is_trainable = param.requires_grad
            num_params = param.numel()
            
            layer_info = {
                'name': name,
                'shape': list(param.shape),
                'params': num_params,
                'trainable': is_trainable
            }
            
            summary['layers'].append(layer_info)
            summary['total_params'] += num_params
            
            if is_trainable:
                summary['trainable_params'] += num_params
            else:
                summary['non_trainable_params'] += num_params
        
        # Get activation functions if possible
        activation_info = self._extract_activation_functions()
        if activation_info:
            summary['activation_functions'] = activation_info
            
        return summary
    
    def _extract_activation_functions(self) -> List[Dict[str, str]]:
        """
        Extract information about activation functions used in the network.
        
        Returns:
            List of dictionaries containing activation function information.
        """
        activations = []
        
        # Try to extract from q_net
        if hasattr(self.agent, 'q_net') and hasattr(self.agent.q_net, 'q_net'):
            q_net = self.agent.q_net.q_net
            
            # If q_net is a sequential model, we can iterate through its modules
            if hasattr(q_net, 'children'):
                for i, module in enumerate(q_net.children()):
                    module_name = module.__class__.__name__
                    if "ReLU" in module_name or "Tanh" in module_name or "Sigmoid" in module_name:
                        activations.append({
                            'layer': f'q_net.q_net.{i}',
                            'type': module_name
                        })
        
        return activations
    
    def extract_model_architecture(self) -> Dict[str, Any]:
        """
        Extract and structure information about the model architecture.
        
        Returns:
            Dictionary containing model architecture information
        """
        architecture = {
            "agent_type": "DQN",
            "observation_space": {
                "type": str(type(self.agent.observation_space)),
                "shape": str(self.agent.observation_space.shape) if hasattr(self.agent.observation_space, "shape") else "None",
                "spaces": {}
            },
            "action_space": {
                "type": str(type(self.agent.action_space)),
                "n": self.agent.action_space.n if hasattr(self.agent.action_space, "n") else "N/A"
            },
            "layers": []
        }
        
        # Add feature extractor information if available
        feature_extractor = self.agent.policy.features_extractor
        if feature_extractor is not None:
            architecture["feature_extractor"] = {
                "type": str(type(feature_extractor)),
                "module_path": f"{feature_extractor.__class__.__module__}.{feature_extractor.__class__.__name__}"
            }
            
            # Add features_dim if available (might be stored in _features_dim)
            if hasattr(feature_extractor, "_features_dim"):
                architecture["feature_extractor"]["features_dim"] = feature_extractor._features_dim
            elif hasattr(feature_extractor, "features_dim"):
                architecture["feature_extractor"]["features_dim"] = feature_extractor.features_dim
                
            # Add CNN and MLP information if available
            if hasattr(feature_extractor, "use_cnn"):
                architecture["feature_extractor"]["use_cnn"] = feature_extractor.use_cnn
            if hasattr(feature_extractor, "use_mlp"):
                architecture["feature_extractor"]["use_mlp"] = feature_extractor.use_mlp
        else:
            architecture["feature_extractor"] = {
                "type": "None",
                "note": "No feature extractor found in this agent"
            }
        
        # If observation space is Dict, add info about each space
        if hasattr(self.agent.observation_space, "spaces"):
            for key, space in self.agent.observation_space.spaces.items():
                architecture["observation_space"]["spaces"][key] = {
                    "type": str(type(space)),
                    "shape": str(space.shape) if hasattr(space, "shape") else "N/A"
                }
        
        # Extract layer information for online network only (exclude target network)
        for name, param in self.agent.policy.state_dict().items():
            # Skip target network parameters
            if 'target' in name:
                continue
                
            if "weight" in name:
                architecture["layers"].append({
                    "name": name,
                    "shape": list(param.shape),
                    "params": int(np.prod(param.shape))
                })
                
        # Add a note about the target network being excluded
        architecture["note"] = "Target network parameters are intentionally excluded as they are only used during training."
        
        return architecture
    
    def save_model_components(self) -> Dict[str, str]:
        """
        Save all model components in formats that are convenient for reloading.
        
        Returns:
            Dictionary with paths to saved model components
        """
        # Create output subdirectory for this agent
        agent_dir = os.path.join(self.output_dir, self.agent_name)
        os.makedirs(agent_dir, exist_ok=True)
        
        saved_paths = {}
        
        # Save architecture info
        architecture = self.extract_model_architecture()
        architecture_path = os.path.join(agent_dir, "architecture.json")
        with open(architecture_path, 'w') as f:
            json.dump(architecture, f, indent=2, cls=NumpyEncoder)
        saved_paths["architecture"] = architecture_path
        
        # Save weight statistics
        weight_stats = self.analyze_weight_statistics()
        weight_stats_path = os.path.join(agent_dir, "weight_statistics.json")
        with open(weight_stats_path, 'w') as f:
            json.dump(weight_stats, f, indent=2, cls=NumpyEncoder)
        saved_paths["weight_statistics"] = weight_stats_path
        
        # Save network summary
        network_summary = self.generate_network_summary()
        network_summary_path = os.path.join(agent_dir, "network_summary.json")
        with open(network_summary_path, 'w') as f:
            json.dump(network_summary, f, indent=2, cls=NumpyEncoder)
        saved_paths["network_summary"] = network_summary_path
        
        # Generate and save visualizations
        try:
            vis_dir = os.path.join(agent_dir, "visualizations")
            os.makedirs(vis_dir, exist_ok=True)
            
            # Generate weight distribution plots
            weight_dist_path = self.visualize_weight_distributions(vis_dir)
            if weight_dist_path:
                saved_paths["weight_distributions"] = weight_dist_path
                
            # Generate network diagram
            network_diagram_path = self.generate_network_diagram(vis_dir)
            if network_diagram_path:
                saved_paths["network_diagram"] = network_diagram_path
        except Exception as e:
            print(f"Warning: Could not generate visualizations: {e}")
        
        # Save the online policy network only (used for inference)
        policy_path = os.path.join(agent_dir, "policy.pt")
        torch.save(self.agent.policy, policy_path)
        saved_paths["policy"] = policy_path
        
        # Save the online Q-network only (not the target network)
        q_net_path = os.path.join(agent_dir, "q_network.pt")
        torch.save(self.agent.q_net, q_net_path)
        saved_paths["q_network"] = q_net_path
        
        # Save feature extractor if available
        feature_extractor = self.get_feature_extractor()
        if feature_extractor is not None:
            feature_extractor_path = os.path.join(agent_dir, "feature_extractor.pt")
            torch.save(feature_extractor, feature_extractor_path)
            saved_paths["feature_extractor"] = feature_extractor_path
        
        # Save state dictionaries for easier weight inspection and editing
        # Only save the online policy network state dict
        policy_state_dict = {k: v for k, v in self.agent.policy.state_dict().items() if 'target' not in k}
        policy_state_dict_path = os.path.join(agent_dir, "policy_state_dict.pt")
        torch.save(policy_state_dict, policy_state_dict_path)
        saved_paths["policy_state_dict"] = policy_state_dict_path
        
        # Only save the online Q-network state dict
        q_net_state_dict = {k: v for k, v in self.agent.q_net.state_dict().items() if 'target' not in k}
        q_net_state_dict_path = os.path.join(agent_dir, "q_network_state_dict.pt")
        torch.save(q_net_state_dict, q_net_state_dict_path)
        saved_paths["q_network_state_dict"] = q_net_state_dict_path
        
        # Create a README with instructions
        readme_path = os.path.join(agent_dir, "README.md")
        with open(readme_path, 'w') as f:
            f.write(f"# Extracted DQN Model: {self.agent_name}\n\n")
            f.write("This directory contains extracted model components that can be used for:\n")
            f.write("1. Model analysis and visualization\n")
            f.write("2. Weight editing and modification\n")
            f.write("3. Inference without the full Stable-Baselines3 environment\n\n")
            f.write("## Files\n\n")
            for key, path in saved_paths.items():
                if isinstance(path, str):  # Only include actual file paths
                    f.write(f"- `{os.path.basename(path)}`: {key}\n")
            f.write("\n## Loading Example\n\n")
            f.write("```python\n")
            f.write("import torch\n\n")
            f.write("# Load entire policy network\n")
            f.write("policy = torch.load('policy.pt')\n\n")
            f.write("# Load state dict for weight inspection/editing\n")
            f.write("state_dict = torch.load('policy_state_dict.pt')\n")
            f.write("# Modify weights as needed\n")
            f.write("# state_dict['some_layer.weight'] = modified_weights\n\n")
            f.write("# Load modified state dict back into model\n")
            f.write("policy.load_state_dict(state_dict)\n")
            f.write("```\n\n")
            f.write("## Note on Target Network\n\n")
            f.write("The target Q-network (q_net_target) has been intentionally excluded from the extracted models ")
            f.write("as it is only used during training for stabilizing updates and is not needed for inference.\n")
            f.write("The extracted models contain only the online network (q_net) which is used for inference and decision-making.\n")
        
        saved_paths["readme"] = readme_path
        
        if self.verbose:
            print(f"Extracted model components saved to {agent_dir}")
            print("Saved files:")
            for key, path in saved_paths.items():
                print(f"  {key}: {os.path.basename(path)}")
            print("\nNote: Target Q-network (used only for training) has been excluded from extraction.")
        
        return saved_paths

    def visualize_weight_distributions(self, output_dir: str) -> Optional[str]:
        """
        Generate visualizations of weight distributions for each layer.
        
        Args:
            output_dir: Directory to save visualizations
            
        Returns:
            Path to saved visualization or None if visualization failed
        """
        try:
            # Create a multi-panel figure for weight distributions
            state_dict = self.agent.q_net.state_dict()
            # Filter out target network weights
            weight_layers = [name for name in state_dict.keys() if 'weight' in name and 'target' not in name]
            
            if not weight_layers:
                print("No weight layers found for visualization")
                return None
                
            n_layers = len(weight_layers)
            n_cols = min(2, n_layers)
            n_rows = (n_layers + n_cols - 1) // n_cols
            
            plt.figure(figsize=(15, 4 * n_rows))
            
            for i, layer_name in enumerate(weight_layers):
                weights = state_dict[layer_name].detach().cpu().numpy().flatten()
                
                plt.subplot(n_rows, n_cols, i + 1)
                sns.histplot(weights, kde=True)
                plt.title(f"Weight Distribution: {layer_name}")
                plt.xlabel("Weight Value")
                plt.ylabel("Frequency")
                plt.grid(alpha=0.3)
            
            plt.tight_layout()
            
            # Save the figure
            output_path = os.path.join(output_dir, "weight_distributions.png")
            plt.savefig(output_path)
            plt.close()
            
            return output_path
        except Exception as e:
            print(f"Error visualizing weight distributions: {e}")
            return None

    def generate_network_diagram(self, output_dir: str) -> Optional[str]:
        """
        Generate a basic network architecture diagram using matplotlib.
        This is a simplified representation as full neural network visualization 
        would require more sophisticated tools.
        
        Args:
            output_dir: Directory to save the diagram
            
        Returns:
            Path to saved diagram or None if generation failed
        """
        try:
            # Extract layer information
            layers = []
            for name, param in self.agent.policy.named_parameters():
                # Skip target network parameters
                if 'target' in name:
                    continue
                    
                if 'weight' in name:
                    # Extract layer information from name (assumes standard format)
                    layer_info = {'name': name, 'shape': param.shape}
                    layers.append(layer_info)
            
            if not layers:
                print("No layers found for diagram generation")
                return None
                
            # Sort layers by name for a more logical flow
            layers.sort(key=lambda x: x['name'])
            
            # Create a simple diagram
            plt.figure(figsize=(12, 8))
            
            # Determine max neurons in any layer for scaling
            max_neurons = max([max(layer['shape']) for layer in layers])
            
            # Draw boxes for each layer
            layer_positions = []
            for i, layer in enumerate(layers):
                width = 1.0
                height = layer['shape'][0] / max_neurons * 5
                x = i * 2
                y = 5 - height/2
                
                layer_positions.append((x, y, width, height))
                
                # Draw the layer box
                rect = plt.Rectangle((x, y), width, height, 
                                    facecolor='lightblue', edgecolor='blue', alpha=0.7)
                plt.gca().add_patch(rect)
                
                # Add layer name
                plt.text(x + width/2, y - 0.5, layer['name'], 
                        ha='center', va='center', fontsize=9, rotation=45)
                
                # Add shape information
                plt.text(x + width/2, y + height/2, f"{layer['shape']}", 
                        ha='center', va='center', fontsize=8)
            
            # Set plot limits
            plt.xlim(-1, len(layers) * 2 + 1)
            plt.ylim(0, 10)
            plt.axis('off')
            plt.title(f"Network Architecture: {self.agent_name} (Online Network Only)")
            
            # Save the diagram
            output_path = os.path.join(output_dir, "network_diagram.png")
            plt.savefig(output_path)
            plt.close()
            
            return output_path
        except Exception as e:
            print(f"Error generating network diagram: {e}")
            return None


def main():
    parser = argparse.ArgumentParser(description='Extract neural network models from DRL agents')
    parser.add_argument('--agent_path', type=str, required=True, 
                        help='Path to the DQN agent zip file')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Directory to save extracted models')
    parser.add_argument('--verbose', action='store_true', default=True,
                        help='Print detailed information')
    
    args = parser.parse_args()
    
    extractor = DQNModelExtractor(
        agent_path=args.agent_path,
        output_dir=args.output_dir,
        verbose=args.verbose
    )
    
    extractor.save_model_components()


if __name__ == "__main__":
    main()
