"""
The vast majority of this architecture originates from Andrej Karpathy
https://github.com/karpathy/pytorch-made/blob/master/made.py
"""

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

class MaskedLinear(nn.Linear):
    """
    Linear layer that allows for a configurable mask on the weights
    """
    
    def __init__(self, in_features, out_features, bias=True):
        super().__init__(in_features, out_features, bias)        
        self.register_buffer('mask', torch.ones(out_features, in_features))
        
    def set_mask(self, mask):
        self.mask.data.copy_(torch.from_numpy(mask.astype(np.uint8).T))
        
    def forward(self, input):
        return F.linear(input, self.mask * self.weight, self.bias)
    
class GMADE(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, num_frames=5, num_masks=1, ordering="forward"):
        """
        input_size: integer; number of inputs
        hidden_sizes: list of integers; number of units in hidden layers
        output_size: integer; number of outputs
        num_frames: integer; number of mel spectrogram frames
        num_masks: integer; number of masks to cycle through
        ordering: str; ordering of the frames (forward, backward, middle)
        """

        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_sizes = hidden_sizes
        assert self.output_size % self.input_size == 0, "output_size must be integer multiple of output_size"

        # Define simple MaskedLinear neural net
        self.net = []
        sizes = [input_size] + hidden_sizes + [output_size]
        for h0, h1 in zip(sizes, sizes[1:]):
            self.net.extend([
                MaskedLinear(h0, h1),
                nn.ReLU(),
            ])
        # Pop last ReLU from the output layer
        self.net.pop()
        self.net = nn.Sequential(*self.net)

        # Number of Mel coefficients
        self.num_frames = num_frames
        self.num_mels = self.input_size // num_frames

        # Seed for orders of the model ensemble
        self.num_masks = num_masks
        self.seed = 0
        # Dictionary for mask ordering
        self.m = {}
        # Build the initial self.m connectivity
        self.update_masks(ordering=ordering)

    def update_masks(self, ordering):
        if self.m and self.num_masks == 1: return
        L = len(self.hidden_sizes)

        # Fetch the next seed and construct a random stream
        rng = np.random.RandomState(self.seed)
        self.seed = (self.seed + 1) % self.num_masks

        # Replicate frame orders for all mels in the frame
        if ordering == "forward":
            expanded_order = np.repeat([0, 1, 2, 3, 4], self.num_mels)
        elif ordering == "backward":
            expanded_order = np.repeat([4, 3, 2, 1, 0], self.num_mels)
        elif ordering == "middle":
            expanded_order = np.repeat([0, 1, 3, 4, 2], self.num_mels)
        else:
            raise ValueError("ordering must be forward, backward or middle")

        # Sample the order of the inputs and the connectivity of all neurons
        # for middle frame ordering
        self.m[-1] = expanded_order
        for l in range(L):
            self.m[l] = rng.randint(self.m[l-1].min(), self.num_frames - 1, size=self.hidden_sizes[l])
        
        # Construct the mask matrices
        masks = [self.m[l-1][:,None] <= self.m[l][None,:] for l in range(L)]
        masks.append(self.m[L-1][:,None] < expanded_order[None,:])

        # Handle the case where output_size = k * input_size (e.g. mu, logvar as output)
        if self.output_size > self.input_size:
            k = int(self.output_size / self.input_size)
            # Replicate mask across the other outputs
            masks[-1] = np.concatenate([masks[-1]] * k, axis=1)

        # Set the masks in all MaskedLinear layers
        layers = [l for l in self.net.modules() if isinstance(l, MaskedLinear)]
        for l, m in zip(layers, masks):
            l.set_mask(m)

    def forward(self, x):
        return self.net(x)