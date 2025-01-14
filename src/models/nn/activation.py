"""Utilities for activation functions."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def Activation(activation=None, size=None, dim=-1):
    if activation in [ None, 'id', 'identity', 'linear', 'none' ]:
        return nn.Identity()
    elif activation == 'tanh':
        return nn.Tanh()
    elif activation == 'relu':
        return nn.ReLU()
    elif activation == 'gelu':
        return nn.GELU()
    elif activation == 'elu':
        return nn.ELU()
    elif activation in ['swish', 'silu']:
        return nn.SiLU()
    elif activation == 'glu':
        return nn.GLU(dim=dim)
    elif activation.startswith('glu-'):
        return GLU(dim=dim, activation=activation[4:])
    elif activation == 'sigmoid':
        return nn.Sigmoid()
    elif activation == 'softplus':
        return nn.Softplus()
    elif activation == 'modrelu':
        return ModReLU(size)
    elif activation in ['sqrelu', 'relu2']:
        return SquaredReLU()
    elif activation == 'laplace':
        return Laplace()
    # r.s.o
    elif activation == 'spiking':
        return F_custom()
    # Earlier experimentation with a LN in the middle of the block instead of activation
    # IIRC ConvNext does something like this?
    # elif activation == 'ln':
    #     return TransposedLN(dim)
    else:
        raise NotImplementedError("hidden activation '{}' is not implemented".format(activation))

class GLU(nn.Module):
    def __init__(self, dim=-1, activation='sigmoid'):
        super().__init__()
        assert not activation.startswith('glu')
        self.dim = dim
        self.activation_fn = Activation(activation)

    def forward(self, x):
        x, g = torch.split(x, x.size(self.dim)//2, dim=self.dim)
        return x * self.activation_fn(g)

class ModReLU(nn.Module):
    # Adapted from https://github.com/Lezcano/expRNN

    def __init__(self, features):
        # For now we just support square layers
        super().__init__()
        self.features = features
        self.b = nn.Parameter(torch.Tensor(self.features))
        self.reset_parameters()

    def reset_parameters(self):
        self.b.data.uniform_(-0.01, 0.01)

    def forward(self, inputs):
        norm = torch.abs(inputs)
        biased_norm = norm + self.b
        magnitude = F.relu(biased_norm)
        phase = torch.sign(inputs)

        return phase * magnitude


class SquaredReLU(nn.Module):
    def forward(self, x):
        # return F.relu(x)**2
        return torch.square(F.relu(x))  # Could this be faster?

def laplace(x, mu=0.707107, sigma=0.282095):
    x = (x - mu).div(sigma * math.sqrt(2.0))
    return 0.5 * (1.0 + torch.erf(x))

class Laplace(nn.Module):
    def __init__(self, mu=0.707107, sigma=0.282095):
        super().__init__()
        self.mu = mu
        self.sigma = sigma

    def forward(self, x):
        return laplace(x, mu=self.mu, sigma=self.sigma)


# r.s.o - custom spike without update potential

import math
class ZIF(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, gama):
        out = (input >= 0).float()
        L = torch.tensor([gama])
        ctx.save_for_backward(input, out, L)
        return out

    # @staticmethod
    # def backward(ctx, grad_output):
    #     (input, out, others) = ctx.saved_tensors
    #     gama = others[0].item()
    #     grad_input = grad_output
    #     tmp = (1 / gama) * (1 / gama) * ((gama - input.abs()).clamp(min=0))
    #     grad_input = grad_input * tmp
    #     return grad_input, None
    
    @staticmethod
    def backward(ctx, grad_output):
        input, out, alpha_t = ctx.saved_tensors
        alpha = alpha_t.item()
        
        # Surrogate gradient using the derivative of arctan-based surrogate:
        #   g'(x) = alpha / [ 2 (1 + ((π/2) * alpha * x)^2) ]
        grad_input = alpha / (2.0 * (1.0 + ((math.pi / 2.0) * alpha * input) ** 2))
        
        # Multiply by the incoming gradient
        grad_input = grad_output * grad_input
        
        # No gradient for alpha (None)
        return grad_input, None

class LIF(nn.Module):
    def __init__(self, T=0, thresh=1.0, tau=1., gama=1.0):
        super(LIF, self).__init__()
        self.act = ZIF.apply       
        # self.thresh = nn.Parameter(torch.tensor([thresh], device='cuda'), requires_grad=False, )
        self.thresh = torch.tensor([thresh], device='cuda', requires_grad=False)
        self.tau = tau
        self.gama = gama

    def forward(self, x, **kwargs):        
        # input [L, BS, feat]
        L = x.size(0)

        # Init mem
        mem = torch.zeros_like(x[0])    # [BS, feat]

        spike_pot = []
        for t in range(L):
            
            mem = self.tau*mem + x[t, ...]

            # mem should be bigger than 0
            # mem = torch.clamp(mem, min=0)
            
            # print(mem[0])

            temp_spike = self.act(mem-self.thresh, self.gama)
            spike = temp_spike * self.thresh # spike [N, C, H, W]

            # print(spike[0])
            
            ### Soft reset ###
            # mem = mem - spike
            ### Hard reset ###
            mem = mem*(1.-spike)

            spike_pot.append(spike) # spike_pot[0].shape [N, C, H, W]

        s = torch.stack(spike_pot,dim=0) # dimension [T, N, C, H, W]  

        return s    # L, BS, feat

class F_custom(nn.Module):
    def __init__(self, thresh=1.0, gama=1.0):
        super(F_custom, self).__init__()
        self.act = ZIF.apply       
        # self.thresh = nn.Parameter(torch.tensor([thresh], device='cuda'), requires_grad=False, )
        self.thresh = torch.tensor([thresh], device='cuda', requires_grad=False)
        self.gama = gama

    def forward(self, x, **kwargs):        
        # Dimension agnostic
        temp_spike = self.act(x-self.thresh, self.gama)
        spike = temp_spike * self.thresh 

        return spike   