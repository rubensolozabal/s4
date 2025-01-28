# Why CIFAR not working?

'''
Train an S4 model on sequential CIFAR10 / sequential MNIST with PyTorch for demonstration purposes.
This code borrows heavily from https://github.com/kuangliu/pytorch-cifar.

This file only depends on the standalone S4 layer
available in /models/s4/

* Train standard sequential CIFAR:
    python -m example
* Train sequential CIFAR grayscale:
    python -m example --grayscale
* Train MNIST:
    python -m example --dataset mnist --d_model 256 --weight_decay 0.0

The `S4Model` class defined in this file provides a simple backbone to train S4 models.
This backbone is a good starting point for many problems, although some tasks (especially generation)
may require using other backbones.

The default CIFAR10 model trained by this file should get
89+% accuracy on the CIFAR10 test set in 80 epochs.

Each epoch takes approximately 7m20s on a T4 GPU (will be much faster on V100 / A100).
'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import os
import argparse

from models.s4.s4 import S4Block as S4  # Can use full version instead of minimal S4D standalone below
# from models.s4.s4d_sSSM import S4D
from s5 import S5, S5Block

from tqdm.auto import tqdm

# Dropout broke in PyTorch 1.11
if tuple(map(int, torch.__version__.split('.')[:2])) == (1, 11):
    print("WARNING: Dropout is bugged in PyTorch 1.11. Results may be worse.")
    dropout_fn = nn.Dropout
if tuple(map(int, torch.__version__.split('.')[:2])) >= (1, 12):
    dropout_fn = nn.Dropout1d
else:
    dropout_fn = nn.Dropout2d



parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
# Optimizer
parser.add_argument('--lr', default=0.01, type=float, help='Learning rate')
parser.add_argument('--weight_decay', default=0.01, type=float, help='Weight decay')
# Scheduler
# parser.add_argument('--patience', default=10, type=float, help='Patience for learning rate scheduler')
parser.add_argument('--epochs', default=100, type=float, help='Training epochs')
# Dataset
parser.add_argument('--dataset', default='cifar10', choices=['mnist', 'cifar10'], type=str, help='Dataset')
parser.add_argument('--permuted', action='store_true', help='Use permuted MNIST')
parser.add_argument('--grayscale', action='store_true', help='Use grayscale CIFAR10')
# Dataloader
parser.add_argument('--num_workers', default=4, type=int, help='Number of workers to use for dataloader')
parser.add_argument('--batch_size', default=64, type=int, help='Batch size')
# Model
parser.add_argument('--n_layers', default=4, type=int, help='Number of layers')
parser.add_argument('--d_model', default=128, type=int, help='Model dimension')
parser.add_argument('--dropout', default=0.1, type=float, help='Dropout')
parser.add_argument('--prenorm', action='store_true', help='Prenorm')
# General
parser.add_argument('--resume', '-r', action='store_true', help='Resume from checkpoint')

args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print(f'==> Preparing {args.dataset} data..')

def split_train_val(train, val_split):
    train_len = int(len(train) * (1.0-val_split))
    train, val = torch.utils.data.random_split(
        train,
        (train_len, len(train) - train_len),
        generator=torch.Generator().manual_seed(42),
    )
    return train, val

if args.dataset == 'cifar10':

    length = 1024
    if args.grayscale:
        transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize(mean=122.6 / 255.0, std=61.0 / 255.0),
            transforms.Lambda(lambda x: x.view(1, 1024).t())
        ])
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            transforms.Lambda(lambda x: x.view(3, 1024).t())
        ])

    # S4 is trained on sequences with no data augmentation!
    transform_train = transform_test = transform

    trainset = torchvision.datasets.CIFAR10(
        root='./data/cifar/', train=True, download=True, transform=transform_train)
    trainset, _ = split_train_val(trainset, val_split=0.1)

    valset = torchvision.datasets.CIFAR10(
        root='./data/cifar/', train=True, download=True, transform=transform_test)
    _, valset = split_train_val(valset, val_split=0.1)

    testset = torchvision.datasets.CIFAR10(
        root='./data/cifar/', train=False, download=True, transform=transform_test)

    d_input = 3 if not args.grayscale else 1
    d_output = 10

elif args.dataset == 'mnist':

    length = 784
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.view(1, 784).t())
    ])
    transform_train = transform_test = transform


    trainset = torchvision.datasets.MNIST(
        root='./data', train=True, download=True, transform=transform_train)
    trainset, _ = split_train_val(trainset, val_split=0.1)

    valset = torchvision.datasets.MNIST(
        root='./data', train=True, download=True, transform=transform_test)
    _, valset = split_train_val(valset, val_split=0.1)

    testset = torchvision.datasets.MNIST(
        root='./data', train=False, download=True, transform=transform_test)
    
    if args.permuted:

        perm = torch.load("permutation.pt").long() # created using torch.randperm(784)

        class psMNIST(Dataset):
            """ Dataset that defines the psMNIST dataset, given the MNIST data and a fixed permutation """

            def __init__(self, mnist, perm):
                self.mnist = mnist # also a torch.data.Dataset object
                self.perm  = perm

            def __len__(self):
                return len(self.mnist)

            def __getitem__(self, idx):
                img, label = self.mnist[idx]
                unrolled = img.reshape(-1)
                permuted = unrolled[self.perm]
                permuted = permuted.reshape(-1, 1)
                return permuted, label
            
        
        trainset = psMNIST(trainset, perm)
        valset   = psMNIST(valset, perm)
        testset  = psMNIST(testset, perm)

    d_input = 1
    d_output = 10
else: 
    raise NotImplementedError

# Dataloaders

trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
valloader = torch.utils.data.DataLoader(
    valset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)


from spikingjelly.activation_based import neuron, surrogate, layer, functional, encoding

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
        # input [L, BS, feat]
        L = x.size(0)

        spike = self.act(x-self.thresh, self.gama)
        # spike = spike * self.thresh # spike [N, C, H, W]


        return spike    # L, BS, feat
    


class S4Model(nn.Module):

    def __init__(
        self,
        d_input,
        d_output=10,
        d_model=256,
        n_layers=4,
        dropout=0.1,
        drop_kernel=0.0,
        prenorm=False,
    ):
        super().__init__()

        self.prenorm = prenorm

        # Linear encoder (d_input = 1 for grayscale and 3 for RGB)
        self.encoder = nn.Linear(d_input, d_model)

        # Spiking Poisson Encoder
        self.pe = encoding.PoissonEncoder()

        # Stack S4 layers as residual blocks
        self.s4_layers = nn.ModuleList()
        self.activations = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        for _ in range(n_layers):
            self.s4_layers.append(
                # S5Block(d_model, d_model, False)
                S5Block(d_model, d_model, bidir= True, block_count =1, liquid = False, degree = 1, ff_mult = 1., glu = True,
                 ff_dropout = dropout, attn_dropout = dropout)
                # S4D(d_model, dropout=dropout, drop_kernel=drop_kernel, transposed=True, lr=min(0.001, args.lr))
            )
            # self.norms.append(nn.LayerNorm(d_model))
            self.norms.append(nn.BatchNorm1d(length))
            self.dropouts.append(dropout_fn(dropout))
            # self.activations.append(nn.GELU())
            self.activations.append(F_custom(thresh=0.1, gama=2.0))
            # self.activations.append(neuron.IFNode_without_membrane_update(surrogate_function=surrogate.ATan(), step_mode='m', backend = 'torch')) # Cuda not available)

        # Linear decoder
        self.decoder = nn.Linear(d_model, d_output)
        # self.f_spike = neuron.IFNode_without_membrane_update(surrogate_function=surrogate.ATan(), step_mode='m', backend = 'torch') # Cuda not available
        # self.f_spike = neuron.IFNode(surrogate_function=surrogate.ATan(), step_mode='m', backend = 'cupy')
        # self.f_spike = LIF_custom(thresh=0.0, gama=1.0)  
        self.f_h = neuron.IFNode(surrogate_function=surrogate.ATan(), step_mode='m', backend = 'cupy')
        self.W_h = nn.Linear(in_features = d_model , out_features = d_model)

        self.mixer = nn.Linear(d_model, d_model)

    def forward(self, x):
        """
        Input x is shape (B, L, d_input)
        """
        x = self.encoder(x)  # (B, L, d_input) -> (B, L, d_model)
        x = x.transpose(-1, -2)  # (B, L, d_model) -> (B, d_model, L)

        # or spike encoding
        # x = self.f_h(self.encoder(x).transpose(-1,-2).permute(2,0,1).contiguous()) # [B,L,d]-->[L,B,d]
        # x = x.permute(1,2,0).contiguous() # [L,B,d]->[B,d,L] 

        # or poisson encoder
        # x = self.pe(x)

        spike_rates = [0.0 for _ in range(len(self.s4_layers))]
        for i, (layer, norm, dropout, act) in enumerate(zip(self.s4_layers, self.norms, self.dropouts, self.activations)):
            # Each iteration of this loop will map (B, d_model, L) -> (B, d_model, L)

            z = x
            # if self.prenorm:
            #     # Prenorm
            #     z = norm(z.transpose(-1, -2)).transpose(-1, -2)

            # Mixer
            # z = self.mixer(z.permute(0,2,1).contiguous()) # [B,d,L]-->[B,L,d]
            # z = z.permute(0,2,1).contiguous() # [B,L,d]->[B,d,L]

            # Apply S4 block: we ignore the state input and output
            z = layer(z.transpose(-1, -2)).transpose(-1, -2) # [B,d,L]

            # Add activation
            # z_spike = act(z.permute(2,0,1).contiguous()) # [B,d,L]-->[L,B,d]
            # z = z_spike.permute(1,2,0).contiguous() # [L,B,d]->[B,d,L]

            # Spike rate
            spike_rate = z.sum() / z.size(0)   # Avg.spiker per batch
            total_slots = z.size(1) * z.size(2) # Batch size * sequence length
            spike_rates[i] = spike_rate / total_slots

            # Dropout on the output of the S4 block
            z = dropout(z)

            # Residual connection
            # x = z + x
            # x = z

            if not self.prenorm:
                # Postnorm
                z = norm(z.transpose(-1, -2)).transpose(-1, -2)

            # Residual connection
            x = z + x
            # x = z

        # Average spike all blocks
        spike_rate = sum(spike_rates) / len(spike_rates)

        x = x.transpose(-1, -2)

        #r.s.o
        # x = self.f_h(self.W_h(x).transpose(-1,-2).permute(2,0,1).contiguous()) # [B,N,C]->[B,C,N]->[N,B,C]
        # x = x.permute(1,0,2).contiguous() # [N,B,C]->[B,N,C]

        # Pooling: average pooling over the sequence length
        x = x.mean(dim=1)

        # Decode the outputs
        x = self.decoder(x)  # (B, d_model) -> (B, d_output)

        return x

# Model
print('==> Building model..')
model = S4Model(
    d_input=d_input,
    d_output=d_output,
    d_model=args.d_model,
    n_layers=args.n_layers,
    dropout=args.dropout,
    prenorm=args.prenorm,
)

model = model.to(device)
if device == 'cuda':
    cudnn.benchmark = True

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.pth')
    model.load_state_dict(checkpoint['model'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

def setup_optimizer(model, lr, weight_decay, epochs):
    """
    S4 requires a specific optimizer setup.

    The S4 layer (A, B, C, dt) parameters typically
    require a smaller learning rate (typically 0.001), with no weight decay.

    The rest of the model can be trained with a higher learning rate (e.g. 0.004, 0.01)
    and weight decay (if desired).
    """

    # All parameters in the model
    all_parameters = list(model.parameters())

    # General parameters don't contain the special _optim key
    params = [p for p in all_parameters if not hasattr(p, "_optim")]

    # Create an optimizer with the general parameters
    optimizer = optim.AdamW(params, lr=lr, weight_decay=weight_decay)

    # Add parameters with special hyperparameters
    hps = [getattr(p, "_optim") for p in all_parameters if hasattr(p, "_optim")]
    hps = [
        dict(s) for s in sorted(list(dict.fromkeys(frozenset(hp.items()) for hp in hps)))
    ]  # Unique dicts
    for hp in hps:
        params = [p for p in all_parameters if getattr(p, "_optim", None) == hp]
        optimizer.add_param_group(
            {"params": params, **hp}
        )

    # Create a lr scheduler
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=patience, factor=0.2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

    # Print optimizer info
    keys = sorted(set([k for hp in hps for k in hp.keys()]))
    for i, g in enumerate(optimizer.param_groups):
        group_hps = {k: g.get(k, None) for k in keys}
        print(' | '.join([
            f"Optimizer group {i}",
            f"{len(g['params'])} tensors",
        ] + [f"{k} {v}" for k, v in group_hps.items()]))

    return optimizer, scheduler

criterion = nn.CrossEntropyLoss()
optimizer, scheduler = setup_optimizer(
    model, lr=args.lr, weight_decay=args.weight_decay, epochs=args.epochs
)

###############################################################################
# Everything after this point is standard PyTorch training!
###############################################################################

# Training
def train():
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    pbar = tqdm(enumerate(trainloader))
    for batch_idx, (inputs, targets) in pbar:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        # Reset
        functional.reset_net(model)

        acc = 100.*correct/total
        pbar.set_description(
            'Batch Idx: (%d/%d) | Loss: %.3f | Acc: %.3f%% (%d/%d)' %
            (batch_idx, len(trainloader), train_loss/(batch_idx+1), acc, correct, total)
        )

    return acc

def eval(epoch, dataloader, checkpoint=False):
    global best_acc
    model.eval()
    eval_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        pbar = tqdm(enumerate(dataloader))
        for batch_idx, (inputs, targets) in pbar:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            eval_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            # Reset
            functional.reset_net(model)

            pbar.set_description(
                'Batch Idx: (%d/%d) | Loss: %.3f | Acc: %.3f%% (%d/%d)' %
                (batch_idx, len(dataloader), eval_loss/(batch_idx+1), 100.*correct/total, correct, total)
            )

    acc = 100.*correct/total
    # Save checkpoint.
    if checkpoint:
        if acc > best_acc:
            state = {
                'model': model.state_dict(),
                'acc': acc,
                'epoch': epoch,
            }
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            torch.save(state, './checkpoint/ckpt.pth')
            best_acc = acc

    return acc

import wandb
# highlight-start
run = wandb.init(
    # Set the project where this run will be logged
    project="sSSM_cifar",
    # Track hyperparameters and run metadata
    config=args,
)

pbar = tqdm(range(start_epoch, args.epochs))
for epoch in pbar:
    train_acc = train()
    val_acc = eval(epoch, valloader, checkpoint=True)
    test_acc = eval(epoch, testloader)
    scheduler.step()

    pbar.set_description('Epoch: %d | Val acc: %1.3f' % (epoch, val_acc))
    wandb.log({"train_acc": train_acc, "val_acc": val_acc, "test_acc": test_acc})
    # print(f"Epoch {epoch} learning rate: {scheduler.get_last_lr()}")

    

