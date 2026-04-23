from modtorch import NN_Model
import torch

static_0 = torch.rand(1, 10)
static_1 = torch.rand(1, 4)

nn_layers = [
    {'add_input': True, 'save': 'static_0', 'encoder': True},
    {'add_input': True, 'save': 'static_1', 'encoder': True},

    {'layer': 'Linear', 'in_features': 10, 'out_features': 4, 'name': 'static_0', 'encoder': True},
    {'layer': 'SiLU', 'save': 'static_0->feat', 'encoder': True},

    {'layer': 'Linear', 'in_features': 4, 'out_features': 2, 'name': 'static_1', 'encoder': True},
    {'layer': 'SiLU', 'save': 'static_1->feat', 'encoder': True},

    {'module': 'custom', 'layer': 'Concatenate', 'dim': 1, 'name_list': ['static_0->feat','static_1->feat'], 'encoder': True},
    {'layer': 'Linear', 'in_features': 6, 'out_features': 6, 'encoder': True},
    {'layer': 'SiLU', 'save': 'encoder_out', 'encoder': True},

    {'layer': 'Linear', 'in_features': 6, 'out_features': 10, 'name': 'encoder_out', 'save': 'static_0->encoder'},
    {'layer': 'Linear', 'in_features': 6, 'out_features': 4, 'name': 'encoder_out', 'save': 'static_1->encoder'},
    
    {'output_list': ['static_0->encoder','static_1->encoder']}
]

NN = NN_Model(nn_layers)
prev = NN([static_0, static_1])
enc = NN.encoder([static_0, static_1])
print(f'Output 1: {prev[0].detach()} - Output 2: {prev[1].detach()} - Encoded: {enc.detach()}')