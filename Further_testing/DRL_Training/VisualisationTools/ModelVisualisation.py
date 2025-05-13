import torch
from stable_baselines3 import DQN
import matplotlib.pyplot as plt
import networkx as nx

def load_agent_and_report_dimensions(agent_path):
    # Load the trained agent
    model = DQN.load(agent_path)

    # Access the policy network
    policy = model.policy

    # Report the dimensions of the neural network layers
    print("Neural Network Architecture:")
    for name, param in policy.state_dict().items():
        if "weight" in name:
            print(f"Layer: {name}, Dimensions: {param.shape}")

def custom_layout(G):
    """
    Create a custom layout for the neural network graph.
    Neurons in each layer are arranged in a vertical line, and layers are arranged horizontally.
    """
    pos = {}
    layer_positions = {}

    # Group nodes by layer based on their names
    for node in G.nodes():
        layer_name = node.split('_out_')[0]
        if layer_name not in layer_positions:
            layer_positions[layer_name] = len(layer_positions)  # Assign horizontal position

    # Assign positions to nodes
    for node in G.nodes():
        layer_name = node.split('_out_')[0]
        neuron_index = int(node.split('_out_')[1])
        x = layer_positions[layer_name]  # Horizontal position based on layer
        y = -neuron_index  # Vertical position based on neuron index
        pos[node] = (x, y)

    return pos

def visualize_neural_network(agent_path):
    # Load the trained agent
    model = DQN.load(agent_path)
    policy = model.policy

    # Create a directed graph for the neural network
    G = nx.DiGraph()

    # Add nodes and edges based on the layers
    previous_layer_size = None
    for name, param in policy.state_dict().items():
        if "weight" in name:
            layer_size = param.shape[0]
            input_size = param.shape[1]

            # Add nodes for the current layer
            for i in range(layer_size):
                G.add_node(f"{name}_out_{i}", weight=param[i].abs().mean().item())

            # Add edges between layers
            if previous_layer_size is not None:
                for i in range(previous_layer_size):
                    for j in range(layer_size):
                        G.add_edge(f"{prev_name}_out_{i}", f"{name}_out_{j}")

            previous_layer_size = layer_size
            prev_name = name

    # Visualize the graph
    fig, ax = plt.subplots(figsize=(16, 12))  # Increased figure size for better clarity
    pos = custom_layout(G)  # Use the custom layout for better visualization
    weights = nx.get_node_attributes(G, 'weight')
    node_colors = [weights[node] for node in G.nodes()]

    # Create a ScalarMappable for the colorbar
    sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=plt.Normalize(vmin=min(node_colors), vmax=max(node_colors)))
    sm.set_array([])  # Required for ScalarMappable

    nx.draw(G, pos, with_labels=False, node_color=node_colors, cmap=plt.cm.viridis, node_size=50, ax=ax)  # Reduced node size
    fig.colorbar(sm, ax=ax, label="Weight Magnitude")
    plt.title("Neural Network Visualization")
    plt.show()

if __name__ == "__main__":
    agent_paths = ["dqn_minigrid_agent.zip", "dqn_minigrid_agent2.zip", "dqn_minigrid_agent_cnn.zip"]
    for path in agent_paths:
        print(f"Analyzing agent: {path}")
        load_agent_and_report_dimensions(path)
        print("-" * 50)

    agent_path = "dqn_minigrid_agent2.zip"
    # visualize_neural_network(agent_path)