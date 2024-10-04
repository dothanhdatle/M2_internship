import math
import torch
from torch import nn

class Gaussian(nn.Module):

    def __init__(self, base_output_dim):
        super().__init__()
        # base_output_dim is how many rv we have (on the output dim)
        self.base_output_dim = base_output_dim
        self.decoder_output_dim = base_output_dim * 2
    
    def forward(self, decoder_output):
        """
        forward pass
        """
        # transforms the decoder output into the real parameters for the Gaussian
        mu = decoder_output[:, :, : self.base_output_dim]
        log_var = decoder_output[:, :, self.base_output_dim : 2 * self.base_output_dim]
        var = torch.exp(log_var)
        #var = torch.ones_like(log_var)

        return mu, var
    
    def llkh(self, x, reduction="mean"):
        """
        compute loglikelihood
        """
        mu, var = self.param
        llkh = -0.5 * (torch.log(2 * torch.tensor(math.pi)) +
                torch.log(var) + (x - mu)**2 / var)
        if reduction == "mean":
            return torch.mean(llkh, dim=1)
        elif reduction == "sum":
            return torch.sum(llkh, dim=1)
        
    def mean(self):
        """
        Compute the mean of the distribution
        """
        mu, var = self.param
        return mu
