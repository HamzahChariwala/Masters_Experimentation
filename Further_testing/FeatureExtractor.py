import gymnasium as gym
import torch as th
import torch.nn as nn
import numpy as np
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gymnasium import spaces
from gymnasium.core import ObservationWrapper


# class CustomCombinedExtractor(BaseFeaturesExtractor):
#     def __init__(self, observation_space: gym.spaces.Dict, features_dim: int = 256):
#         # Calculate the total feature dimension after processing all modalities
#         super(CustomCombinedExtractor, self).__init__(observation_space, features_dim)
        
#         # Extract the shapes of individual observation components
#         image_shape = observation_space.spaces['CNN_input'].shape
#         print(f"IMAGE_SHAPE: {image_shape}")
#         vector_shape = observation_space.spaces['MLP_input'].shape

#         # Define CNN for image processing
#         self.cnn = nn.Sequential(
#             nn.Conv2d(image_shape[0], 32, kernel_size=8, stride=4),
#             nn.ReLU(),
#             nn.Conv2d(32, 64, kernel_size=4, stride=2),
#             nn.ReLU(),
#             nn.Flatten()
#         )

#         # Compute the output size of the CNN
#         with th.no_grad():
#             sample_input = th.as_tensor(observation_space.spaces['CNN_input'].sample()[None]).float()
#             cnn_output_dim = self.cnn(sample_input).shape[1]

#         # Define MLP for vector processing
#         self.mlp = nn.Sequential(
#             nn.Linear(vector_shape[0], 64),
#             nn.ReLU()
#         )

#         # Total features dimension is the sum of CNN and MLP outputs
#         self._features_dim = cnn_output_dim + 64

#     def forward(self, observations):
#         # Process image and vector inputs separately
#         image_features = self.cnn(observations['CNN_input'])
#         vector_features = self.mlp(observations['MLP_input'])
#         # Concatenate the features
#         return th.cat([image_features, vector_features], dim=1)


# class CustomCombinedExtractor(BaseFeaturesExtractor):
#     def __init__(
#         self,
#         observation_space: gym.spaces.Dict,
#         features_dim: int = 256,
#         cnn_num_layers: int = 2,
#         cnn_channels: list = None,
#         cnn_kernels: list = None,
#         cnn_strides: list = None,
#         cnn_paddings: list = None,  # ✅ NEW
#         mlp_num_layers: int = 1,
#         mlp_hidden_sizes: list = None,
#     ):
#         super(CustomCombinedExtractor, self).__init__(observation_space, features_dim)

#         image_shape = observation_space.spaces['CNN_input'].shape  # (C, H, W)
#         vector_shape = observation_space.spaces['MLP_input'].shape  # (D,)
#         print(f"IMAGE_SHAPE (C, H, W): {image_shape}")
#         print(f"VECTOR_SHAPE: {vector_shape}")

#         # ==== CNN Configuration ====
#         if cnn_channels is None:
#             cnn_channels = [32 * (2**i) for i in range(cnn_num_layers)]
#         if cnn_kernels is None:
#             cnn_kernels = [3 for _ in range(cnn_num_layers)]
#         if cnn_strides is None:
#             cnn_strides = [2 for _ in range(cnn_num_layers)]
#         if cnn_paddings is None:
#             cnn_paddings = [0 for _ in range(cnn_num_layers)]  # default to no padding

#         assert cnn_num_layers == len(cnn_channels) == len(cnn_kernels) == len(cnn_strides) == len(cnn_paddings), \
#             "CNN layer counts and parameter lists must match."

#         # Build CNN
#         cnn_layers = []
#         in_channels = image_shape[0]
#         for out_channels, kernel, stride, padding in zip(cnn_channels, cnn_kernels, cnn_strides, cnn_paddings):
#             cnn_layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=kernel, stride=stride, padding=padding))
#             cnn_layers.append(nn.ReLU())
#             in_channels = out_channels
#         cnn_layers.append(nn.Flatten())
#         self.cnn = nn.Sequential(*cnn_layers)

#         # Compute CNN output shape
#         with th.no_grad():
#             sample_input = th.as_tensor(observation_space.spaces['CNN_input'].sample()[None]).float()
#             cnn_output_dim = self.cnn(sample_input).shape[1]
#         print(f"CNN_OUTPUT_DIM: {cnn_output_dim}")

#         # ==== MLP Configuration ====
#         if mlp_hidden_sizes is None:
#             mlp_hidden_sizes = [64 for _ in range(mlp_num_layers)]

#         assert len(mlp_hidden_sizes) == mlp_num_layers, "MLP layer count and hidden sizes must match."

#         mlp_layers = []
#         last_dim = vector_shape[0]
#         for hidden_size in mlp_hidden_sizes:
#             mlp_layers.append(nn.Linear(last_dim, hidden_size))
#             mlp_layers.append(nn.ReLU())
#             last_dim = hidden_size
#         self.mlp = nn.Sequential(*mlp_layers)
#         print(f"MLP_OUTPUT_DIM: {last_dim}")

#         self._features_dim = cnn_output_dim + last_dim
#         print(f"TOTAL_FEATURE_DIM (CNN + MLP): {self._features_dim}")

#     def forward(self, observations):
#         image_features = self.cnn(observations['CNN_input'])
#         vector_features = self.mlp(observations['MLP_input'])
#         return th.cat([image_features, vector_features], dim=1)
    

class CustomCombinedExtractor(BaseFeaturesExtractor):
    def __init__(
        self,
        observation_space: gym.spaces.Dict,
        features_dim: int = 256,
        cnn_num_layers: int = 2,
        cnn_channels: list = None,
        cnn_kernels: list = None,
        cnn_strides: list = None,
        cnn_paddings: list = None,
        mlp_num_layers: int = 1,
        mlp_hidden_sizes: list = None,
    ):
        super(CustomCombinedExtractor, self).__init__(observation_space, features_dim)

        self.use_cnn = 'CNN_input' in observation_space.spaces
        self.use_mlp = 'MLP_input' in observation_space.spaces

        cnn_output_dim = 0
        mlp_output_dim = 0

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

        self._features_dim = cnn_output_dim + mlp_output_dim
        print(f"TOTAL_FEATURE_DIM (CNN + MLP): {self._features_dim}")

    def forward(self, observations):
        features = []

        if self.use_cnn:
            features.append(self.cnn(observations['CNN_input']))

        if self.use_mlp:
            features.append(self.mlp(observations['MLP_input']))

        return th.cat(features, dim=1) if len(features) > 1 else features[0]




# class SelectiveObservationWrapper(ObservationWrapper):
#     def __init__(self, env, cnn_keys=None, mlp_keys=None):
#         super().__init__(env)

#         # Default to empty lists if no keys are provided
#         self.cnn_keys = cnn_keys or []
#         self.mlp_keys = mlp_keys or []

#         # Ensure the base observation space is a Dict
#         assert isinstance(env.observation_space, spaces.Dict), "Base observation space must be a Dict."

#         # Extract the original observation space
#         base_spaces = env.observation_space.spaces

#         # Define the new observation space with only the keys we want
#         new_spaces = {}

#         # Determine the shape and dtype for CNN_input
#         if self.cnn_keys:
#             missing_cnn_keys = [key for key in self.cnn_keys if key not in base_spaces]
#             if missing_cnn_keys:
#                 raise KeyError(f"The following CNN keys are missing from the observation space: {missing_cnn_keys}")
#             cnn_shapes = [base_spaces[key].shape for key in self.cnn_keys]
#             cnn_dtypes = [base_spaces[key].dtype for key in self.cnn_keys]
#             cnn_shape = (int(len(self.cnn_keys)),) + tuple(int(i) for i in cnn_shapes[0])
#             cnn_dtype = cnn_dtypes[0]
#             new_spaces['CNN_input'] = spaces.Box(low=0, high=255, shape=cnn_shape, dtype=cnn_dtype)

#         # Determine the shape and dtype for MLP_input
#         if self.mlp_keys:
#             missing_mlp_keys = [key for key in self.mlp_keys if key not in base_spaces]
#             if missing_mlp_keys:
#                 raise KeyError(f"The following MLP keys are missing from the observation space: {missing_mlp_keys}")
#             mlp_dim = int(sum(np.prod(base_spaces[key].shape) for key in self.mlp_keys))
#             mlp_dtype = np.float32
#             new_spaces['MLP_input'] = spaces.Box(low=-np.inf, high=np.inf, shape=(mlp_dim,), dtype=mlp_dtype)

#         # Always include log_data for full observation info (for debugging or logging)
#         self.observation_space = spaces.Dict(new_spaces)

#     def observation(self, obs):
#         new_obs = {}

#         # Construct CNN_input
#         if self.cnn_keys:
#             try:
#                 cnn_inputs = [obs[key] for key in self.cnn_keys]
#                 new_obs['CNN_input'] = np.stack(cnn_inputs, axis=0)
#             except KeyError as e:
#                 raise KeyError(f"Missing key in observation for CNN_input: {e}")

#         # Construct MLP_input
#         if self.mlp_keys:
#             try:
#                 mlp_inputs = [np.ravel(obs[key]) for key in self.mlp_keys]
#                 new_obs['MLP_input'] = np.concatenate(mlp_inputs, axis=0)
#             except KeyError as e:
#                 raise KeyError(f"Missing key in observation for MLP_input: {e}")

#         # Store full original observation in log_data
#         new_obs['log_data'] = dict(obs)  # <-- SB3 ignores this at runtime

#         return new_obs
    

class SelectiveObservationWrapper(ObservationWrapper):
    def __init__(self, env, cnn_keys=None, mlp_keys=None):
        super().__init__(env)

        self.cnn_keys = cnn_keys or []
        self.mlp_keys = mlp_keys or []
        self.latest_log_data = {}

        assert isinstance(env.observation_space, spaces.Dict)

        base_spaces = env.observation_space.spaces
        new_spaces = {}

        # CNN input
        if self.cnn_keys:
            # Assume all cnn_keys have the same (H, W, C) shape
            shape_hwc = base_spaces[self.cnn_keys[0]].shape  # (H, W, C)
            c = shape_hwc[2] * len(self.cnn_keys)
            h, w = shape_hwc[0], shape_hwc[1]
            new_spaces['CNN_input'] = spaces.Box(low=0, high=255, shape=(c, h, w), dtype=np.uint8)

        # MLP input
        if self.mlp_keys:
            mlp_dim = int(sum(np.prod(base_spaces[key].shape) for key in self.mlp_keys))
            new_spaces['MLP_input'] = spaces.Box(low=-np.inf, high=np.inf, shape=(mlp_dim,), dtype=np.float32)

        self.observation_space = spaces.Dict(new_spaces)

    def observation(self, obs):
        new_obs = {}

        # CNN input formatting: (H, W, C) → (C, H, W)
        if self.cnn_keys:
            # cnn_inputs = [np.transpose(obs[key], (2, 0, 1)) for key in self.cnn_keys]  # each → (C, H, W)
            cnn_inputs = [obs[key] for key in self.cnn_keys]  # already in (C, H, W)
            new_obs['CNN_input'] = np.concatenate(cnn_inputs, axis=0)  # → (C_total, H, W)

        # MLP input: flatten each input
        if self.mlp_keys:
            mlp_inputs = [np.ravel(obs[key]) for key in self.mlp_keys]
            new_obs['MLP_input'] = np.concatenate(mlp_inputs, axis=0)

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

