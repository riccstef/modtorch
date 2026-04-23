from . import modlib

def Reshape(l):
    return modlib.Reshape(l['shape'])

def Transpose(l):
    return modlib.Transpose(l['dim0'], l['dim1'])

def Concatenate(l):
    return modlib.Concatenate(l.get('dim', 1))

def Add(l):
    return modlib.Add()

def Diff(l):
    return modlib.Diff()

def Multiply(l):
    return modlib.Multiply()

def CumSum(l):
    return modlib.CumSum(l.get('dim',1))

def ScalarAddProd(l):
    x_add = l.get('x_add', 0.)
    x_prod = l.get('x_prod', 1.)
    return modlib.ScalarAddProd(x_add=x_add, x_prod=x_prod)

def ReduceMean(l):
    return modlib.ReduceMean(l['dim'], keepdim=l.get('keepdim',True))

def ReduceMax(l):
    return modlib.ReduceMax(l['dim'], keepdim=l.get('keepdim',True))

def ReduceStd(l):
    correction = l.get('correction', 1)
    return modlib.ReduceStd(l['dim'], correction=correction, keepdim=l.get('keepdim',True))

def Split(l):
    return modlib.Split(l['indices'], l.get('dim',1))

def ListSelect(l):
    return modlib.ListSelect(l['ind'])

def GLU(l):
    return modlib.GLU(l['input_size'])

def FastGLU(l):
    return modlib.FastGLU(l['input_size'])

def CBAM(l):
    return modlib.CBAM(l['channels'], l['reduction'], l['activation'], l['kernel_size'])

def TSMixer(l):
    n_ts = l.get('n_ts',None)
    n_hidden = l.get('n_hidden',None)
    n_mixer = l.get('n_mixer',1)
    activation = l.get('activation','linear')
    dropout = l.get('dropout', 0.0)
    dropout_fm = l.get('dropout_fm', dropout)
    normalization = l.get('normalization','BatchNorm')
    output_hidden = l.get('output_hidden', False)
    causal = l.get('causal', False)
    return modlib.TSMixer(l['n_lag'], l['n_features'], l['n_output'], n_ts=n_ts, n_hidden=n_hidden, n_mixer=n_mixer, activation=activation,
        dropout=dropout, dropout_fm=dropout_fm, normalization=normalization, output_hidden=output_hidden, causal=causal)

def TSMixerExt(l):
    # simplified version of original TSMixerExt with only static features and separate mixer for static features
    # which can be used for time series forecasting with static covariates
    n_ts = l.get('n_ts',None)
    n_hidden = l.get('n_hidden',None)
    n_hidden_static = l.get('n_hidden_static',None)
    n_mixer = l.get('n_mixer',1)
    n_static_mixer = l.get('n_static_mixer',1)
    activation = l.get('activation','linear')
    activation_mixing_stage = l.get('activation_mixing_stage', None)
    dropout = l.get('dropout', 0.0)
    dropout_fm = l.get('dropout_fm', dropout)
    dropout_sm = l.get('dropout_sm', dropout)
    normalization = l.get('normalization','BatchNorm')
    output_hidden = l.get('output_hidden', False)
    causal = l.get('causal', False)
    return modlib.TSMixerExt(l['n_lag'], l['n_features'], l['n_static'], l['n_output'], n_ts=n_ts, n_hidden=n_hidden,
        n_hidden_static=n_hidden_static, n_mixer=n_mixer, n_static_mixer=n_static_mixer, activation=activation,
        activation_mixing_stage=activation_mixing_stage, dropout=dropout, dropout_fm=dropout_fm, dropout_sm=dropout_sm, normalization=normalization,
        output_hidden=output_hidden, causal=causal)

def GCBlock(l):
    return modlib.GCBlock(l['channels'], l['reduction'], l['activation'])

def LSTM_Sequence(l):
    return modlib.LSTM_Sequence(l['input_size'], l['hidden_size'], num_layers=l.get('num_layers', 1), batch_first=l.get('batch_first', True),
        dropout=l.get('dropout', 0.0), bidirectional=l.get('bidirectional', False))

def LSTM_Last(l):
    return modlib.LSTM_Last(l['input_size'], l['hidden_size'], num_layers=l.get('num_layers', 1), batch_first=l.get('batch_first', True),
        dropout=l.get('dropout', 0.0), bidirectional=l.get('bidirectional', False))

def GhostBatchNorm2d(l):
    return modlib.GhostBatchNorm2d(l['num_features'], l['ghost_batch_size'])
 
def MixConv2d(l):
    stride = l.get('stride',1)
    dilation = l.get('dilation',1)
    bias = l.get('bias',False)
    return modlib.MixConv2d(l['in_channels'], l['out_channels'], l['kernel_sizes'], stride, dilation, bias, l['split'])

def MixConv2dGLU(l):
    stride = l.get('stride',1)
    dilation = l.get('dilation',1)
    bias = l.get('bias',False)
    split = l.get('bias',True)
    normalization = l.get('normalization','BatchNorm2d') # 'BatchNorm2d', 'GroupNorm', None
    ghost_batch_size = l.get('ghost_batch_size',32)
    num_groups = l.get('num_groups',l['out_channels']) # InstanceNorm
    return modlib.MixConv2dGLU(l['in_channels'], l['out_channels'], l['kernel_sizes'], stride, dilation, bias, split, normalization,
        ghost_batch_size=ghost_batch_size, num_groups=num_groups)

def Conv2dGLU(l):
    stride = l.get('stride',1)
    padding = l.get('padding',0)
    dilation = l.get('dilation',1)
    groups = l.get('groups',1)
    bias = l.get('bias',False)
    normalization = l.get('normalization','BatchNorm2d') # 'BatchNorm2d', 'GroupNorm', None
    padding_mode = l.get('padding_mode','zeros')
    ghost_batch_size = l.get('ghost_batch_size',32)
    num_groups = l.get('num_groups',l['out_channels']) # InstanceNorm
    return modlib.Conv2dGLU(l['in_channels'], l['out_channels'], l['kernel_size'], stride, padding, dilation, groups, bias, normalization,
        padding_mode, ghost_batch_size=ghost_batch_size, num_groups=num_groups)

def GaussianNoise(l):
    return modlib.GaussianNoise(l.get('stddev', 1.0))