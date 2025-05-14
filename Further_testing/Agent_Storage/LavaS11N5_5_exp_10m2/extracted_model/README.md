# Extracted DQN Model: agent

This directory contains extracted model components that can be used for:
1. Model analysis and visualization
2. Weight editing and modification
3. Inference without the full Stable-Baselines3 environment

## Files

- `architecture.json`: architecture
- `weight_statistics.json`: weight_statistics
- `network_summary.json`: network_summary
- `weight_distributions.png`: weight_distributions
- `network_diagram.png`: network_diagram
- `policy.pt`: policy
- `q_network.pt`: q_network
- `policy_state_dict.pt`: policy_state_dict
- `q_network_state_dict.pt`: q_network_state_dict

## Loading Example

```python
import torch

# Load entire policy network
policy = torch.load('policy.pt')

# Load state dict for weight inspection/editing
state_dict = torch.load('policy_state_dict.pt')
# Modify weights as needed
# state_dict['some_layer.weight'] = modified_weights

# Load modified state dict back into model
policy.load_state_dict(state_dict)
```

## Note on Target Network

The target Q-network (q_net_target) has been intentionally excluded from the extracted models as it is only used during training for stabilizing updates and is not needed for inference.
The extracted models contain only the online network (q_net) which is used for inference and decision-making.
