import torch
from torch import nn
import torch.nn.functional as F

class GLU(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.linear = nn.Linear(input_size, input_size)
        self.linear_sigma = nn.Linear(input_size, input_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.linear_sigma(x)) * self.linear(x)

class FastGLU(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.linear = nn.Linear(input_size, input_size*2)
        self.glu = nn.GLU()

    def forward(self, x):
        return self.glu(self.linear(x))

def get_activation(activation, **kwargs):
    activation = activation.lower()
    if activation == 'tanh':
        return nn.Tanh()
    elif activation == 'relu':
        return nn.ReLU()
    elif activation == 'silu':
        return nn.SiLU()
    elif activation == 'glu':
        return GLU(kwargs['input_size'])
    elif activation == 'fastglu':
        return FastGLU(kwargs['input_size'])
    elif activation == 'linear':
        return nn.Identity()
    else:
        raise ValueError(f"Unknown activation: {activation}")

class Reshape(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(x.size(0), *self.shape)  

class Transpose(nn.Module):
    def __init__(self, dim0, dim1):
        super().__init__()
        self.dim0 = dim0
        self.dim1 = dim1

    def forward(self, x):
        return x.transpose(self.dim0, self.dim1)  

class Concatenate(nn.Module):
    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return torch.cat(x, dim=self.dim)

class Add(nn.Module):
    def forward(self, x):
        return sum(x)

class Diff(nn.Module):
    def forward(self, x):
        return x[0]-x[1]

class ReduceMean(nn.Module):
    def __init__(self, dim, keepdim=True):
        super().__init__()
        self.dim = dim
        self.keepdim = keepdim

    def forward(self, x):
        return x.mean(dim=self.dim, keepdim=self.keepdim)

class ReduceMax(nn.Module):
    def __init__(self, dim, keepdim=True):
        super().__init__()
        self.dim = dim
        self.keepdim = keepdim

    def forward(self, x):
        return x.max(dim=self.dim, keepdim=self.keepdim).values

class ReduceStd(nn.Module):
    def __init__(self, dim, correction=1, keepdim=True):
        super().__init__()
        self.dim = dim
        self.correction = correction
        self.keepdim = keepdim

    def forward(self, x):
        return x.std(dim=self.dim, correction=self.correction, keepdim=self.keepdim)

class Multiply(nn.Module):
    def forward(self, x_list):
        out = x_list[0]
        for x in x_list[1:]:
            out = out * x
        return out

class ScalarAddProd(nn.Module):
    def __init__(self, x_add=0., x_prod=1.):
        super().__init__()
        self.x_add = x_add
        self.x_prod = x_prod

    def forward(self, x):
        return self.x_add + self.x_prod * x

class CumSum(nn.Module):
    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return torch.cumsum(x, dim=self.dim)

class Split(nn.Module):
    def __init__(self, ind, dim):
        super().__init__()
        if not isinstance(ind, list):
            ind = list(ind)
        self.ind = ind
        self.dim = dim

    def forward(self, x: torch.Tensor):
        return torch.tensor_split(x, self.ind, self.dim)
    
class ListSelect(nn.Module):
    def __init__(self, ind):
        super().__init__()
        if not isinstance(ind, int):
            raise Exception("ind must be a single integer")
        self.ind = ind

    def forward(self, x: torch.Tensor):
        return x[self.ind]

class CBAM(nn.Module):
    def __init__(self, channels, reduction, activation, kernel_size):
        super().__init__()
        ch_reduction = max(1, channels//reduction)
        self.activation = get_activation(activation, input_size=ch_reduction)
        # Channel Attention
        self.mlp = nn.Sequential(
            nn.Linear(channels, ch_reduction),
            self.activation,
            nn.Linear(ch_reduction, channels))
        # Spatial Attention
        self.spatial = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size//2, bias=False)

    def forward(self, x):
        # ----- Channel attention -----
        avg_pool = F.adaptive_avg_pool2d(x, 1).squeeze(-1).squeeze(-1)
        max_pool = F.adaptive_max_pool2d(x, 1).squeeze(-1).squeeze(-1)
        ch_att = self.mlp(avg_pool) + self.mlp(max_pool)
        ch_att = torch.sigmoid(ch_att).unsqueeze(-1).unsqueeze(-1)
        x = x*ch_att
        # ----- Spatial attention -----
        spatial_att = torch.cat([torch.mean(x, dim=1, keepdim=True), torch.max(x, dim=1, keepdim=True)[0]], dim=1)
        spatial_att = torch.sigmoid(self.spatial(spatial_att))
        x = x*spatial_att
        return x

class TSMixer(nn.Module):
    def __init__(self, n_lag, n_features, n_output, n_ts=None, n_hidden=None, n_mixer=1, activation='fastglu', dropout=0.0, dropout_fm=None,
        normalization='BatchNorm', output_hidden=False, causal=False):
        super().__init__()
        if n_hidden is None:
            n_hidden = n_features
        if n_ts is None:
            n_ts = n_lag
        if dropout_fm is None:
            dropout_fm = dropout
        self.n_mixer = n_mixer
        self.mixer = nn.ModuleList([
            _Mixer(n_lag, n_features, n_ts, n_hidden, n_output, activation, dropout, dropout_fm, normalization, causal) if n_mixer==1
            else _Mixer(n_lag, n_features, n_ts, n_hidden, n_hidden, activation, dropout, dropout_fm, normalization, causal) if i==0
            else _Mixer(n_ts, n_hidden, n_ts, n_hidden, n_output, activation, dropout, dropout_fm, normalization, causal) if i==n_mixer-1
            else _Mixer(n_ts, n_hidden, n_ts, n_hidden, n_hidden, activation, dropout, dropout_fm, normalization, causal) for i in range(n_mixer)])
        self.tp = _MixerTP(n_ts)
        self.output_hidden = output_hidden

    def forward(self, x):
        for i in range(self.n_mixer):
            x = self.mixer[i](x)
        if self.output_hidden:
            x_hidden = x
        x = self.tp(x)
        if self.output_hidden:
            return x, x_hidden
        else:
            return x

class _Mixer(nn.Module):
    def __init__(self, n_lag, n_features, n_ts, n_hidden, n_output, activation, dropout, dropout_fm, normalization, causal):
        super().__init__()
        if n_output != n_features:
            self.linear_res_f = nn.Linear(n_features, n_output)
        else:
            self.linear_res_f = nn.Identity()
        if n_ts != n_lag:
            self.linear_res_t = nn.Linear(n_lag, n_ts)
        else:
            self.linear_res_t = nn.Identity()
        self.activation_tm = get_activation(activation, input_size=n_ts)
        self.activation_fm = get_activation(activation, input_size=n_hidden)
        if dropout > 0:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = nn.Identity()
        if dropout_fm > 0:
            self.dropout_fm = nn.Dropout(dropout_fm)
        else:
            self.dropout_fm = nn.Identity()
        if normalization == 'BatchNorm':
            self.norm_tm = nn.BatchNorm2d(1)
            self.norm_fm = nn.BatchNorm2d(1)
        elif normalization == 'LayerNorm':
            self.norm_tm = nn.LayerNorm(n_features)
            self.norm_fm = nn.LayerNorm(n_features)
        elif normalization is None:
            self.norm_tm = nn.Identity()
            self.norm_fm = nn.Identity()
        else:
            raise Exception(f"Normalizzazione {normalization} non disponibile\n")
        if not causal:
            self.linear_tm = nn.Linear(n_lag, n_ts)
        elif causal and n_lag == n_ts:
            self.linear_tm = _CausalLinear(n_lag)
        else:
            raise Exception("n_lag and n_ts must be with causal=True")
        self.linear_fm_1 = nn.Linear(n_features, n_hidden)
        self.linear_fm_2 = nn.Linear(n_hidden, n_output)

    def forward(self, x):
        x_tm = self.norm_tm(x.unsqueeze(1)).squeeze(1)
        x_tm = self.dropout(self.activation_tm(self.linear_tm(x_tm.transpose(1,2)))).transpose(1,2)
        x_tm = x_tm + self.linear_res_t(x.transpose(1,2)).transpose(1,2) # residual
        x_fm = self.norm_fm(x_tm.unsqueeze(1)).squeeze(1)
        x_fm = self.dropout_fm(self.linear_fm_2(self.dropout_fm(self.activation_fm(self.linear_fm_1(x_fm)))))
        x_fm = x_fm + self.linear_res_f(x_tm) # residual
        return x_fm

class _CausalLinear(nn.Module):
    def __init__(self, T):
        super().__init__()
        self.T = T
        self.theta = nn.Parameter(torch.randn(T*(T+1)//2))
        self.bias = nn.Parameter(torch.zeros(T))
        self.register_buffer("tril_idx", torch.tril_indices(T, T))

    def forward(self, x):
        W = torch.zeros(self.T, self.T, device=x.device)
        W[self.tril_idx[0], self.tril_idx[1]] = self.theta
        y = torch.matmul(x, W.T) + self.bias
        return y
    
class _MixerTP(nn.Module):
    def __init__(self, n_lag):
        super().__init__()
        self.linear_tp = nn.Linear(n_lag, 1)

    def forward(self, x):
        x_tp = self.linear_tp(x.transpose(1,2)).squeeze(2)
        return x_tp

class GCBlock(nn.Module):
    def __init__(self, channels, reduction, activation):
        super().__init__()
        ch_reduction = max(1, channels//reduction)
        self.conv_wk = nn.Conv2d(channels, 1, kernel_size=1, bias=True)
        self.fc_v1 = nn.Linear(channels, ch_reduction, bias=False)
        self.ln = nn.LayerNorm(ch_reduction)
        self.activation = get_activation(activation, input_size=ch_reduction)
        self.fc_v2 = nn.Linear(ch_reduction, channels)

    def forward(self, x):
        B, C, H, W = x.shape
        N = H * W
        attn = self.conv_wk(x).view(B, 1, N)
        alpha = F.softmax(attn, dim=2)
        context = torch.bmm(x.view(B, C, N), alpha.transpose(1, 2)).view(B,C)
        v = self.fc_v2(self.activation(self.ln(self.fc_v1(context))))
        out = x + v.view(B, C, 1, 1)
        return out

