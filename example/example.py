from modtorch import NN_Model
import torch

ts = torch.rand(1, 20, 6)
img = torch.rand(1, 3, 64, 64)
static = torch.rand(1, 10)

nn_layers = [
    {'add_input': True, 'save': 'time_series'},
    {'add_input': True, 'save': 'image'},
    {'add_input': True, 'save': 'static'},

    {'layer': 'Linear', 'in_features': 6, 'out_features': 4, 'name': 'time_series'},
    {'layer': 'LayerNorm', 'normalized_shape': 4, 'save': 'time_series->proj'},

    {'layer': 'Conv2d', 'in_channels': 3, 'out_channels': 6, 'kernel_size': 3, 'padding': 1, 'bias': False, 'name': 'image'},
    {'layer': 'BatchNorm2d', 'num_features': 6},
    {'layer': 'SiLU'},
    {'layer': 'StochasticDepth', 'p': 0.1, 'save': 'image->feat'},

    {'layer': 'AvgPool2d', 'kernel_size': 3, 'stride': 1, 'padding': 1, 'name': 'image'},
    {'layer': 'Conv2d', 'in_channels': 3, 'out_channels': 6, 'kernel_size': 1, 'bias': False},
    {'layer': 'BatchNorm2d', 'num_features': 6, 'save': 'image->res'},
    {'module': 'custom', 'layer': 'Add', 'name_list': ['image->feat','image->res']},
    {'module': 'custom', 'layer': 'GCBlock', 'channels': 6, 'reduction': 4, 'activation': 'SiLU'},
    {'layer': 'AdaptiveAvgPool2d', 'output_size': 1},
    {'layer': 'Flatten', 'start_dim': 2, 'end_dim': 3},
    {'module': 'custom', 'layer': 'Transpose', 'dim0': 2, 'dim1': 1},
    {'layer': 'Linear', 'in_features': 6, 'out_features': 4, 'save': 'image->feat'},
    {'module': 'custom', 'layer': 'Multiply', 'name_list': ['time_series->proj','image->feat'], 'save': 'ts+image'},

    {'module': 'custom', 'layer': 'TSMixer', 'n_lag': 20, 'n_features': 4, 'n_output': 4, 'n_mixer': 1,
        'activation': 'fastglu', 'dropout': 0.1, 'normalization': 'BatchNorm', 'name': 'ts+image', 'save': 'ts_mixer->out'},

    {'layer': 'Linear', 'in_features': 10, 'out_features': 4, 'name': 'static'},
    {'layer': 'LayerNorm', 'normalized_shape': 4},
    {'layer': 'Softmax', 'dim': 1, 'save': 'gate'},
    {'module': 'custom', 'layer': 'Multiply', 'name_list': ['ts_mixer->out','gate']},
    {'module': 'custom', 'layer': 'Split', 'indices': [2], 'dim': 1, 'save': ['out1','out2']},
    {'output_list': ['out1','out2', 'gate']}
]

NN = NN_Model(nn_layers)
prev = NN([ts, img, static])
print(f'Output 1: {prev[0].detach()} - Output 2: {prev[1].detach()} - Gate: {prev[2].detach()}')