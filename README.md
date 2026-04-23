# MODTORCH

MODTORCH is a powerful meta-language for building PyTorch networks on the fly, without having to write custom PyTorch classes manually.

This is currently a beta version. Contributions, suggestions, and forks are welcome.

MOE model creation is already working, but the documentation is still in progress.

## Overview

In MODTORCH, a network is defined as a list of dictionaries. Each dictionary represents a layer or a tensor operation.

This makes it possible to define complex architectures dynamically, including custom tensor manipulations, multiple inputs, saved intermediate tensors, and reusable named outputs.

## Rules

Each dictionary in the list describes one step of the network.

- Each dictionary can contain the name of the module where the layer is defined.
- If omitted, the default is `'module': 'basic'`, which refers to `basic.py` and includes standard PyTorch layers.
- You can define your own custom layers (see `modlib.py`) and register them in MODTORCH (see `custom.py`).
- To use custom layers, set `'module': 'custom'` or another registered module name.

- Each dictionary can also contain the name of the layer to execute.
- If omitted, the default is `'layer': 'Identity'`.
- Any additional key-value pairs in the dictionary are passed as arguments to the selected layer.
- For standard PyTorch layers, use the same argument names as in PyTorch.

## Inputs

The model definition must begin with layers used to load the inputs.

Given a list of input tensors, you can load them in two ways:

- Use `'add_input': True` to load inputs sequentially.
- Use `'sel_input': N` to select the input at position `N` in the input list.

## Saving and reusing tensors

- Use `'save': 'xyz'` to store the input or output of a layer under the name `'xyz'`.
- You can also save lists when needed.
- Use `'name': 'xyz'` to load a previously saved tensor and use it as the input of the current layer.
- If `'name'` is not specified, the output of the previous layer is used.
- If a layer requires multiple inputs, use `'name_list': ['abc', 'xyz']`.

## Custom flags

You can add custom flags to any dictionary for later use during training or post-processing.

For example:

- `'encoder': True` can be used to mark layers involved in the encoder path.
- These flags can then be used by helper methods or custom workflows.

## Basic example

```python
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
```

## Encoder example

```python
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

    {'module': 'custom', 'layer': 'Concatenate', 'dim': 1, 'name_list': ['static_0->feat', 'static_1->feat'], 'encoder': True},
    {'layer': 'Linear', 'in_features': 6, 'out_features': 6, 'encoder': True},
    {'layer': 'SiLU', 'save': 'encoder_out', 'encoder': True},

    {'layer': 'Linear', 'in_features': 6, 'out_features': 10, 'name': 'encoder_out', 'save': 'static_0->encoder'},
    {'layer': 'Linear', 'in_features': 6, 'out_features': 4, 'name': 'encoder_out', 'save': 'static_1->encoder'},

    {'output_list': ['static_0->encoder', 'static_1->encoder']}
]

NN = NN_Model(nn_layers)
prev = NN([static_0, static_1])
enc = NN.encoder([static_0, static_1])

print(
    f'Output 1: {prev.detach()} - '
    f'Output 2: {prev.detach()} - '
    f'Encoded: {enc.detach()}'
)
```

## Notes

- MODTORCH is designed for flexibility and rapid experimentation.
- It is especially useful when you want to define architectures dynamically from configuration files or Python dictionaries.
- Custom modules and tensor routing make it possible to build more advanced architectures without writing dedicated PyTorch model classes.

## Project status

MODTORCH is still under active development.

Some features, such as MOE model creation, are already functional but not yet fully documented.

Contributions, issues, and forks are welcome.
