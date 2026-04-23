from modtorch import NN_Model
import torch

static = torch.rand(1, 10)

nn_layers = [
    {'add_input': True},
    {'layer': 'Linear', 'in_features': 10, 'out_features': 10},
    {'layer': 'LayerNorm', 'normalized_shape': 10},
    {'layer': 'SiLU'},
    {'layer': 'Linear', 'in_features': 10, 'out_features': 1},
]

NN = NN_Model(nn_layers)
prev = NN([static])
print(f'Output: {prev.detach()}')