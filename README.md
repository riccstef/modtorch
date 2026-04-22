# MODTORCH

MODTORCH is a powerful meta-language to build PyTorch networks on the fly without having to code PyTorch classes. This a beta version. Contributes and forks are welcome.
The MOE model creation is working but still to be documented.

To build a network model MODTORCH uses a list of dictionaries. Every one is a network layer. Some of them permit custom manipulations of tensors to increment flexibility of the meta-language.

## RULES

- Every dictionary should contain the name of the module where the layer is. Could be omitted, default is 'module': 'basic' which refers to basic.py with some standard PyTorch layers inside. You can write your owm custom layers (see modlib.py) and link them in MODTORCH (see custom.py). To use custom layers you must add 'module': 'custom' (or a different link module name).

- Every dictionary should contain a layer name to be executed. Could be omitted, default is 'layer': 'Identity'.
