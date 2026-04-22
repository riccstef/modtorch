import modlib

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

def GCBlock(l):
    return modlib.GCBlock(l['channels'], l['reduction'], l['activation'])
