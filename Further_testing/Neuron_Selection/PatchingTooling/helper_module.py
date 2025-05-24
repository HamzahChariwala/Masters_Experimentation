import torch
from torch import nn

class NeuronPatcher:
    def __init__(self, model, clean_acts, patch_indices):
        """
        model:         nn.Module (e.g. model.policy.q_net)
        clean_acts:    dict[layer_name] = Tensor shape (1, n_neurons)
        patch_indices: dict[layer_name] = list of neuron indices to force
        """
        self.clean_acts     = clean_acts
        self.patch_indices  = patch_indices
        self.orig_forward   = {}   # store original forwards

        # Walk the submodules and patch only layers we care about
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear) and name in patch_indices:
                self._patch(name, module)

    def _patch(self, name, module):
        # Save original
        self.orig_forward[name] = module.forward

        # Grab the clean activations for this layer & indices
        clean_vals = self.clean_acts[name]              # Tensor (1, C)
        idxs       = self.patch_indices[name]           # e.g. [5, 17]

        def patched_forward(x):
            # Compute original linear output
            out = self.orig_forward[name](x)
            # Replace only the selected neuron(s)
            # out shape: (batch, C). We assume batch=1 here; adapt if larger.
            out = out.clone()  # avoid in-place on autograd buffer
            for i in idxs:
                out[:, i] = clean_vals[:, i]
            return out

        # Monkey-patch
        module.forward = patched_forward

    def restore(self, model):
        # Restore each module's forward
        for name, module in model.named_modules():
            if name in self.orig_forward:
                module.forward = self.orig_forward[name]
