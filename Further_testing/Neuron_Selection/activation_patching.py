from stable_baselines3 import DQN
import numpy as np
import torch
from Neuron_Selection.PatchingTooling.helper_module import NeuronPatcher

# 1) Load your agent
agent = DQN.load("dqn_model.zip", device="cpu")
q_net = agent.policy.q_net

# 2) Load clean activations saved earlier
clean_data = np.load("clean_acts.npz")      # arrays named raw_fc1, raw_fc2, ...
clean_acts = {
    name.replace("raw_", ""): torch.from_numpy(arr)
    for name, arr in clean_data.items()
}

# 3) Decide which neurons to patch
patch_indices = {
    "mlp_extractor.policy_net.2": [3],   # e.g. layer 'fc2', neuron 3
    "mlp_extractor.policy_net.4": [0,1]  # e.g. layer 'fc3', neurons 0 and 1
}

# 4) Run corrupted pass to get its baseline
corr_obs = np.load("corrupted_obs.npy")
corr_obs = torch.tensor(corr_obs, dtype=torch.float32).unsqueeze(0)
with torch.no_grad():
    corrupt_out = q_net(corr_obs).clone()

# 5) Patch specific neurons
patcher = NeuronPatcher(q_net, clean_acts, patch_indices)
with torch.no_grad():
    patched_out = q_net(corr_obs).clone()

# 6) Restore original forwards
patcher.restore(q_net)

# 7) Compare
print("Corrupted action:", corrupt_out.argmax().item())
print(" Patched action:",   patched_out.argmax().item())
