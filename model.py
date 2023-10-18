import torch
import torch.nn as nn
import torch.nn.functional as F

# Taken from https://github.com/e-hulten/made/blob/master/models/made.py
class MaskedLinear(nn.Linear):
    """Linear transformation with masked out elements. y = x.dot(mask*W.T) + b"""

    def __init__(self, n_in: int, n_out: int, bias: bool = True) -> None:
        """
        Args:
            n_in: Size of each input sample.
            n_out:Size of each output sample.
            bias: Whether to include additive bias. Default: True.
        """
        super().__init__(n_in, n_out, bias)
        self.mask = None

    def _initialise_mask(self, mask: torch.Tensor):
        """Internal method to initialise mask."""
        self.mask = mask.to()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply masked linear transformation."""
        return F.linear(x, self.mask * self.weight, self.bias)

class GMADE(nn.Module):
    def __init__(self, hidden_size, device=torch.device("cpu")):
        super(GMADE, self).__init__()
        
        # Input to hidden layer using MaskedLinear
        self.i2h = MaskedLinear(128 * 5, hidden_size).to(device)
        
        # Hidden layer to mu and sigma
        self.h2mu = nn.Linear(hidden_size, 1).to(device)
        self.h2sigma = nn.Linear(hidden_size, 1).to(device)

        # Create mask for i2h layer
        self.mask = torch.ones_like(self.i2h.weight).to(device)
        
        # Get indices for middle frame 
        middle_frame_indices = list(range(2 * 128, 3 * 128))
        
        # Mask out middle frame weights
        self.mask[:, middle_frame_indices] = 0

        # Initialize mask in MaskedLinear layer
        self.i2h._initialise_mask(self.mask)

    def forward(self, x):
        x = F.relu(self.i2h(x))
        
        mu = self.h2mu(x)
        sigma = self.h2sigma(x)
        
        return mu, sigma