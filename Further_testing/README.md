# VAE for State-Action Pair Encoding

A variational autoencoder (VAE) implementation for encoding state-action pairs into latent vectors to identify similar behavioral patterns.

## Setup

1. Install dependencies:
   ```
   pip install torch numpy matplotlib scikit-learn tqdm pyyaml
   ```

2. Configure parameters in `vae_config.yml`

## Preparing Your Dataset

This implementation expects a dataset of state-action pairs. You should structure your data as:

```python
# Example of preparing your data
import pickle
import numpy as np

# Your collected data should have these two arrays
states = np.array([...])  # Shape: (n_samples, state_dim)
actions = np.array([...])  # Shape: (n_samples, action_dim) or (n_samples,) for discrete actions

# Save as pickle
with open('my_dataset.pkl', 'wb') as f:
    pickle.dump({'states': states, 'actions': actions}, f)
```

The `StateActionDataset` class handles:
- Converting NumPy arrays to PyTorch tensors
- Enforcing correct data types (float32)
- Providing batch access during training
- Creating a uniform interface for different data formats

## Usage

### Basic Training

```bash
python vae_state_action.py
```

### Using a Custom Config

```bash
python vae_state_action.py path/to/custom_config.yml
```

## Configuration

Key parameters in `vae_config.yml`:

- `data_path`: Path to your pre-collected data file (pickle format)
- `data_format`: Format of your data file (currently supports "pickle")
- `latent_dim`: Dimension of the latent space
- `epochs`: Number of training epochs
- `visualize`: Whether to generate a t-SNE visualization

## Creating Custom Datasets

For advanced use cases, you can create datasets directly:

```python
from vae_state_action import StateActionDataset
import torch
from torch.utils.data import DataLoader

# Create dataset from your data
dataset = StateActionDataset(states, actions)

# Create dataloader for batch processing
dataloader = DataLoader(
    dataset, 
    batch_size=128, 
    shuffle=True,
    num_workers=2
)

# Use dataloader for training
for batch in dataloader:
    states_batch = batch['state']  # Shape: (batch_size, state_dim)
    actions_batch = batch['action']  # Shape: (batch_size, action_dim)
    # Process batch...
```

## Key Functions

- `train_vae(vae, dataloader, optimizer, device, epochs, beta)`: Train the VAE model
- `visualize_latent_space(vae, dataloader, device, perplexity)`: Generate t-SNE visualization
- `vae.get_latent(state, action)`: Encode a state-action pair into latent space

## Model Access

After training:
1. Load model: `vae, _ = load_model(vae, optimizer, 'vae_output/vae_model.pth')`
2. Encode states/actions: `latent = vae.get_latent(state_tensor, action_tensor)`
3. Use latent vectors for classification or clustering 