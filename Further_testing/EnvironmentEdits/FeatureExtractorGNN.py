import gymnasium as gym
import torch as th
import torch.nn as nn
import numpy as np
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gymnasium import spaces
from gymnasium.core import ObservationWrapper


class GNNLayer(nn.Module):
    """Simple Graph Neural Network layer.
    
    This is a basic GNN layer that performs message passing between nodes
    in a graph. It can be customized based on specific requirements.
    """
    def __init__(self, in_features, out_features):
        super(GNNLayer, self).__init__()
        self.W = nn.Linear(in_features, out_features, bias=False)
        self.W_self = nn.Linear(in_features, out_features, bias=True)
        
    def forward(self, x, adj):
        """
        x: Node features (batch_size, num_nodes, in_features)
        adj: Adjacency matrix (batch_size, num_nodes, num_nodes)
        """
        # Message passing: multiply node features by adjacency
        # This aggregates messages from neighboring nodes
        neighbor_messages = th.bmm(adj, x)  # (batch_size, num_nodes, in_features)
        
        # Apply transformations
        neighbor_transformed = self.W(neighbor_messages)  # (batch_size, num_nodes, out_features)
        self_transformed = self.W_self(x)  # (batch_size, num_nodes, out_features)
        
        # Combine self and neighbor features and apply non-linearity
        return th.relu(neighbor_transformed + self_transformed)


class CustomCombinedExtractorGNN(BaseFeaturesExtractor):
    def __init__(
        self,
        observation_space: gym.spaces.Dict,
        features_dim: int = 256,
        # CNN parameters
        cnn_num_layers: int = 2,
        cnn_channels: list = None,
        cnn_kernels: list = None,
        cnn_strides: list = None,
        cnn_paddings: list = None,
        # MLP parameters
        mlp_num_layers: int = 1,
        mlp_hidden_sizes: list = None,
        # GNN parameters
        gnn_num_layers: int = 2,
        gnn_hidden_sizes: list = None,
    ):
        super(CustomCombinedExtractorGNN, self).__init__(observation_space, features_dim)

        # Check which inputs are available
        self.use_cnn = 'CNN_input' in observation_space.spaces
        self.use_mlp = 'MLP_input' in observation_space.spaces
        self.use_gnn = 'GNN_nodes' in observation_space.spaces and 'GNN_adjacency' in observation_space.spaces

        cnn_output_dim = 0
        mlp_output_dim = 0
        gnn_output_dim = 0

        # === CNN Setup ===
        if self.use_cnn:
            image_shape = observation_space.spaces['CNN_input'].shape
            print(f"IMAGE_SHAPE (C, H, W): {image_shape}")
            if cnn_channels is None:
                cnn_channels = [32 * (2**i) for i in range(cnn_num_layers)]
            if cnn_kernels is None:
                cnn_kernels = [3 for _ in range(cnn_num_layers)]
            if cnn_strides is None:
                cnn_strides = [2 for _ in range(cnn_num_layers)]
            if cnn_paddings is None:
                cnn_paddings = [0 for _ in range(cnn_num_layers)]

            assert cnn_num_layers == len(cnn_channels) == len(cnn_kernels) == len(cnn_strides) == len(cnn_paddings), \
                "CNN config lists must match cnn_num_layers"

            cnn_layers = []
            in_channels = image_shape[0]
            for out_channels, kernel, stride, padding in zip(cnn_channels, cnn_kernels, cnn_strides, cnn_paddings):
                cnn_layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=kernel, stride=stride, padding=padding))
                cnn_layers.append(nn.ReLU())
                in_channels = out_channels
            cnn_layers.append(nn.Flatten())
            self.cnn = nn.Sequential(*cnn_layers)

            with th.no_grad():
                sample_input = th.as_tensor(observation_space.spaces['CNN_input'].sample()[None]).float()
                cnn_output_dim = self.cnn(sample_input).shape[1]
            print(f"CNN_OUTPUT_DIM: {cnn_output_dim}")
        else:
            self.cnn = None

        # === MLP Setup ===
        if self.use_mlp:
            vector_shape = observation_space.spaces['MLP_input'].shape
            print(f"VECTOR_SHAPE: {vector_shape}")
            if mlp_hidden_sizes is None:
                mlp_hidden_sizes = [64 for _ in range(mlp_num_layers)]

            assert len(mlp_hidden_sizes) == mlp_num_layers, "MLP config must match mlp_num_layers"

            mlp_layers = []
            last_dim = vector_shape[0]
            for hidden_size in mlp_hidden_sizes:
                mlp_layers.append(nn.Linear(last_dim, hidden_size))
                mlp_layers.append(nn.ReLU())
                last_dim = hidden_size
            self.mlp = nn.Sequential(*mlp_layers)
            mlp_output_dim = last_dim
            print(f"MLP_OUTPUT_DIM: {mlp_output_dim}")
        else:
            self.mlp = None
            
        # === GNN Setup ===
        if self.use_gnn:
            # Get node feature dimensions
            node_shape = observation_space.spaces['GNN_nodes'].shape
            num_nodes, node_features = node_shape[0], node_shape[1]
            print(f"GNN_NODE_SHAPE: (nodes={num_nodes}, features={node_features})")
            
            if gnn_hidden_sizes is None:
                gnn_hidden_sizes = [64 for _ in range(gnn_num_layers)]

            assert len(gnn_hidden_sizes) == gnn_num_layers, "GNN config must match gnn_num_layers"
            
            # Create GNN layers
            self.gnn_layers = nn.ModuleList()
            in_features = node_features
            
            for hidden_size in gnn_hidden_sizes:
                self.gnn_layers.append(GNNLayer(in_features, hidden_size))
                in_features = hidden_size
                
            # Pooling layer to reduce GNN output to a fixed size
            # Using a simple global mean pooling approach
            self.gnn_pooling = lambda x: th.mean(x, dim=1)  # (batch, nodes, features) -> (batch, features)
            
            # Final projection for GNN
            self.gnn_projection = nn.Linear(in_features, in_features)
            
            # GNN output dimension after pooling
            gnn_output_dim = in_features
            print(f"GNN_OUTPUT_DIM: {gnn_output_dim}")
        else:
            self.gnn_layers = None
            self.gnn_pooling = None
            self.gnn_projection = None

        # Total features dimension
        self._features_dim = cnn_output_dim + mlp_output_dim + gnn_output_dim
        print(f"TOTAL_FEATURE_DIM (CNN + MLP + GNN): {self._features_dim}")

    def forward(self, observations):
        features = []

        if self.use_cnn and 'CNN_input' in observations:
            features.append(self.cnn(observations['CNN_input']))

        if self.use_mlp and 'MLP_input' in observations:
            features.append(self.mlp(observations['MLP_input']))
            
        if self.use_gnn and 'GNN_nodes' in observations and 'GNN_adjacency' in observations:
            # Extract node features and adjacency matrix
            nodes = observations['GNN_nodes']  # (batch, num_nodes, node_features)
            adj = observations['GNN_adjacency']  # (batch, num_nodes, num_nodes)
            
            # Apply GNN layers
            x = nodes
            for gnn_layer in self.gnn_layers:
                x = gnn_layer(x, adj)
                
            # Apply pooling to get a graph-level representation
            x = self.gnn_pooling(x)  # (batch, hidden_size)
            
            # Final projection
            x = self.gnn_projection(x)
            
            features.append(x)

        return th.cat(features, dim=1) if len(features) > 1 else features[0]


class SelectiveObservationWrapperGNN(ObservationWrapper):
    def __init__(self, env, cnn_keys=None, mlp_keys=None, gnn_node_keys=None, gnn_edge_keys=None):
        super().__init__(env)

        self.cnn_keys = cnn_keys or []
        self.mlp_keys = mlp_keys or []
        self.gnn_node_keys = gnn_node_keys or []
        self.gnn_edge_keys = gnn_edge_keys or []
        self.latest_log_data = {}

        assert isinstance(env.observation_space, spaces.Dict)

        base_spaces = env.observation_space.spaces
        new_spaces = {}

        # CNN input
        if self.cnn_keys:
            # Assume all cnn_keys have the same (H, W, C) shape, or in our case (C, H, W)
            shape_chw = base_spaces[self.cnn_keys[0]].shape  # (C, H, W)
            c = shape_chw[0] * len(self.cnn_keys)
            h, w = shape_chw[1], shape_chw[2]
            new_spaces['CNN_input'] = spaces.Box(low=0, high=255, shape=(c, h, w), dtype=np.uint8)

        # MLP input
        if self.mlp_keys:
            mlp_dim = int(sum(np.prod(base_spaces[key].shape) for key in self.mlp_keys))
            new_spaces['MLP_input'] = spaces.Box(low=-np.inf, high=np.inf, shape=(mlp_dim,), dtype=np.float32)
            
        # GNN inputs
        if self.gnn_node_keys:
            # For GNN, we need node features and adjacency matrix
            # First, determine the total number of nodes and features per node
            total_nodes = sum(base_spaces[key].shape[0] for key in self.gnn_node_keys)
            node_features = base_spaces[self.gnn_node_keys[0]].shape[1] if len(self.gnn_node_keys) > 0 else 0
            
            # Create spaces for node features and adjacency
            new_spaces['GNN_nodes'] = spaces.Box(
                low=-np.inf, high=np.inf, 
                shape=(total_nodes, node_features), 
                dtype=np.float32
            )
            
            # Adjacency matrix space - square matrix with size = total_nodes
            new_spaces['GNN_adjacency'] = spaces.Box(
                low=0, high=1,
                shape=(total_nodes, total_nodes),
                dtype=np.float32
            )

        self.observation_space = spaces.Dict(new_spaces)

    def observation(self, obs):
        new_obs = {}

        # CNN input formatting
        if self.cnn_keys:
            cnn_inputs = [obs[key] for key in self.cnn_keys]  # Each should be (C, H, W)
            new_obs['CNN_input'] = np.concatenate(cnn_inputs, axis=0)  # → (C_total, H, W)

        # MLP input: flatten each input
        if self.mlp_keys:
            mlp_inputs = [np.ravel(obs[key]) for key in self.mlp_keys]
            new_obs['MLP_input'] = np.concatenate(mlp_inputs, axis=0)
            
        # GNN input: node features and adjacency matrix
        if self.gnn_node_keys:
            # Process node features
            node_features = [obs[key] for key in self.gnn_node_keys]
            if node_features:
                new_obs['GNN_nodes'] = np.vstack(node_features)  # Stack node features vertically
                
                # Create adjacency matrix
                # This is just a placeholder - you'll need to define how to create your adjacency matrix
                total_nodes = new_obs['GNN_nodes'].shape[0]
                
                if self.gnn_edge_keys:
                    # If edge information is provided, use it to build the adjacency matrix
                    adj_matrix = np.zeros((total_nodes, total_nodes), dtype=np.float32)
                    
                    # Example: edge_key might contain (source, target, weight) tuples
                    for edge_key in self.gnn_edge_keys:
                        if edge_key in obs:
                            edge_data = obs[edge_key]
                            # Process edge data to populate adjacency matrix
                            # This is just an example - adjust based on your edge representation
                            for edge in edge_data:
                                src, dst, weight = edge
                                adj_matrix[src, dst] = weight
                    
                    new_obs['GNN_adjacency'] = adj_matrix
                else:
                    # If no edge information, create a default adjacency matrix
                    # For demonstration, creating a fully connected graph
                    # Replace this with your actual graph structure logic
                    adj_matrix = np.ones((total_nodes, total_nodes), dtype=np.float32)
                    np.fill_diagonal(adj_matrix, 0)  # No self-loops
                    new_obs['GNN_adjacency'] = adj_matrix

        # Save original full obs for logging
        self.latest_log_data = dict(obs)

        return new_obs

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        obs = self.observation(obs)
        info['log_data'] = self.latest_log_data
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        obs = self.observation(obs)
        info['log_data'] = self.latest_log_data
        return obs, info 