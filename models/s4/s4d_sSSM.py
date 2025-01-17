"""Minimal version of S4D with extra options and features stripped out, for pedagogical purposes."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

from src.models.nn import DropoutNd

class S4DKernel(nn.Module):
    """Generate convolution kernel from diagonal SSM parameters."""

    def __init__(self, d_model, N=64, dt_min=0.001, dt_max=0.1, lr=None):
        super().__init__()
        # Generate dt
        H = d_model
        log_dt = torch.rand(H) * (
            math.log(dt_max) - math.log(dt_min)
        ) + math.log(dt_min)

        C = torch.randn(H, N // 2, dtype=torch.cfloat)
        self.C = nn.Parameter(torch.view_as_real(C))

        #r.s.o
        # C = torch.ones(H, N // 2, dtype=torch.cfloat)
        # self.C = torch.view_as_real(C).to("cuda")

        self.register("log_dt", log_dt, lr)

        log_A_real = torch.log(0.5 * torch.ones(H, N//2))
        A_imag = math.pi * repeat(torch.arange(N//2), 'n -> h n', h=H)
        self.register("log_A_real", log_A_real, lr)
        self.register("A_imag", A_imag, lr)

    def forward(self, L):
        """
        returns: (..., c, L) where c is number of channels (default 1)
        """

        # Materialize parameters
        dt = torch.exp(self.log_dt) # (H)
        C = torch.view_as_complex(self.C) # (H N)
        A = -torch.exp(self.log_A_real) + 1j * self.A_imag # (H N)

        # Vandermonde multiplication
        dtA = A * dt.unsqueeze(-1)  # (H N)
        K = dtA.unsqueeze(-1) * torch.arange(L, device=A.device) # (H N L)
        #C = C * (torch.exp(dtA)-1.) / A  # B_bar = (torch.exp(dtA)-1.) / A    C = (B_t * C)
        B = (torch.exp(dtA)-1.) / A  # (H N)
        K = 2 * torch.einsum('hnl, hn -> hnl', torch.exp(K), B).real # (H N L)

        return K

    def register(self, name, tensor, lr=None):
        """Register a tensor with a configurable learning rate and 0 weight decay"""

        if lr == 0.0:
            self.register_buffer(name, tensor)
        else:
            self.register_parameter(name, nn.Parameter(tensor))

            optim = {"weight_decay": 0.0}
            if lr is not None: optim["lr"] = lr
            setattr(getattr(self, name), "_optim", optim)


from spikingjelly.activation_based import neuron, surrogate, layer, functional


# --------------------
# Custom GLU function
# --------------------
class custom_glu(nn.Module):
    """
    Custom implementation of Gated Linear Unit (GLU).

    Args:
        x (torch.Tensor): Input tensor of shape (..., 2*d, ...)
                          The dimension that will be split must have size 2*d.
        dim (int): The dimension along which we split the tensor.

    Returns:
        torch.Tensor: Output of the GLU operation.
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        # Split the tensor into two equal halves along 'dim'
        a, b = x.chunk(2, dim=self.dim)
        # Compute the GLU output: a * sigmoid(b)
        out = a * torch.sigmoid(b)
        return out



class S4D(nn.Module):
    def __init__(self, d_model, d_state=256, dropout=0.0, transposed=True, **kernel_args):
        super().__init__()

        self.h = d_model
        self.n = d_state
        self.d_output = self.h
        self.transposed = transposed

        self.D = nn.Parameter(torch.randn(self.h))

        # SSM Kernel
        self.kernel = S4DKernel(self.h, N=self.n, **kernel_args)

        # Pointwise
        self.activation = nn.GELU()
        # dropout_fn = nn.Dropout2d # NOTE: bugged in PyTorch 1.11
        dropout_fn = DropoutNd
        self.dropout = dropout_fn(dropout) if dropout > 0.0 else nn.Identity()

        # position-wise output transform to mix features
        self.output_linear = nn.Sequential(
            nn.Conv1d(self.h, 2*self.h, kernel_size=1),
            # nn.GLU(dim=-2),
            custom_glu(dim=-2),
        )
        # self.f_spike = neuron.IFNode_without_membrane_update(surrogate_function=surrogate.ATan(), step_mode='m', backend = 'cupy')

    def forward(self, u, **kwargs): # absorbs return_output and transformer src mask
        """ Input and output shape (B, H, L) """
        if not self.transposed: u = u.transpose(-1, -2)
        L = u.size(-1)

        # Compute SSM Kernel
        k = self.kernel(L=L) # (H N L)

        # Take kernel 0 
        k = k[0] # (N L)

        # Convolution
        k_f = torch.fft.rfft(k, n=2*L) # (H L)
        u_f = torch.fft.rfft(u, n=2*L) # (B H L)
        y = torch.fft.irfft(u_f*k_f, n=2*L)[..., :L] # (B H L)


        # Spiking neuron [B,L,d] -> [L,B,d]
        # y = self.f_spike(y.transpose(0,2))
        # y = y.transpose(0,2) # [L,B,d] -> [B,L,d]


        # Compute D term in state space equation - essentially a skip connection
        # y = y + u * self.D.unsqueeze(-1)

        # y = self.dropout(self.activation(y))
        # y = self.output_linear(y)

        if not self.transposed: y = y.transpose(-1, -2)
        return y, None # Return a dummy state to satisfy this repo's interface, but this can be modified
