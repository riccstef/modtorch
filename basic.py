from torch import nn
import torchvision

def Identity(l):
    return nn.Identity()

def Flatten(l):
    start_dim = l.get('start_dim',1)
    end_dim = l.get('end_dim',-1)
    return nn.Flatten(start_dim=start_dim, end_dim=end_dim)

def Unflatten(l):
    return nn.Unflatten(dim=l['dim'], unflattened_size=l['unflattened_size'])

def Linear(l):
    x = nn.Linear(l['in_features'], l['out_features'], bias=l.get('bias', True))
    gain_label = l.get('gain', None)
    if gain_label in ('tanh', 'sigmoid', 'relu', 'linear', 'conv1d', 'conv2d', 'conv3d', 'selu', 'leaky_relu'):
        if gain_label == 'leaky_relu':
            negative_slope = l.get('negative_slope', 0.01)
            gain = nn.init.calculate_gain('leaky_relu', negative_slope)
        else:
            gain = nn.init.calculate_gain(gain_label)
    else:
        gain = 1.0
    nn.init.xavier_uniform_(x.weight, gain=gain)
    if x.bias is not None:
        nn.init.zeros_(x.bias)
    return x

def Conv2d(l):
    stride = l.get('stride',1)
    padding = l.get('padding',0)
    dilation = l.get('dilation',1)
    groups = l.get('groups',1)
    bias = l.get('bias',False)
    padding_mode = l.get('padding_mode','zeros')
    return nn.Conv2d(l['in_channels'], l['out_channels'], l['kernel_size'], stride, padding, dilation, groups, bias, padding_mode)

def BatchNorm1d(l):
    return nn.BatchNorm1d(l['num_features'])

def BatchNorm2d(l):
    return nn.BatchNorm2d(l['num_features'])

def GroupNorm(l):
    return nn.GroupNorm(l['num_groups'], l['num_channels'])

def LayerNorm(l):
    return nn.LayerNorm(normalized_shape=l['normalized_shape'])

def Dropout(l):
    return nn.Dropout(p=l['p'])

def Dropout2d(l):
    return nn.Dropout2d(p=l['p'])

def ReLU(l):
    return nn.ReLU()

def SiLU(l):
    return nn.SiLU()

def Tanh(l):
    return nn.Tanh()

def Sigmoid(l):
    return nn.Sigmoid()

def Softmax(l):
    return nn.Softmax(dim=l.get('dim',-1))

def LogSoftmax(l):
    return nn.LogSoftmax(dim=l.get('dim',-1))

def Softplus(l):
    beta = l.get('beta',1)
    return nn.Softplus(beta=beta)

def AvgPool1d(l):
    kernel_size = l['kernel_size']
    stride = l.get('stride',None)
    padding = l.get('padding',0)
    return nn.AvgPool1d(kernel_size, stride, padding)

def AvgPool2d(l):
    stride = l.get('stride',None)
    padding = l.get('padding',0)
    return nn.AvgPool2d(l['kernel_size'], stride, padding)

def MaxPool1d(l):
    kernel_size = l['kernel_size']
    stride = l.get('stride',None)
    padding = l.get('padding',0)
    return nn.MaxPool1d(kernel_size, stride, padding)

def MaxPool2d(l):
    stride = l.get('stride',None)
    padding = l.get('padding',0)
    return nn.MaxPool2d(l['kernel_size'], stride, padding)

def AdaptiveAvgPool2d(l):
    return nn.AdaptiveAvgPool2d(l['output_size'])

def StochasticDepth(l):
    return torchvision.ops.StochasticDepth(p=l['p'], mode=l.get('mode','batch'))
