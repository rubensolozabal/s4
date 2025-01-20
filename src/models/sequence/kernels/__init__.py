from .kernel import ConvKernel, EMAKernel
from .ssm import SSMKernelDense, SSMKernelReal, SSMKernelDiag, SSMKernelDPLR, SSMKernelDiag_snn

registry = {
    'conv': ConvKernel,
    'ema': EMAKernel,
    'dense': SSMKernelDense,
    'slow': SSMKernelDense,
    'real': SSMKernelReal,
    's4d': SSMKernelDiag,
    'diag': SSMKernelDiag,
    's4': SSMKernelDPLR,
    'nplr': SSMKernelDPLR,
    'dplr': SSMKernelDPLR,
    'diagsnn' : SSMKernelDiag_snn, #r.s.o
}
