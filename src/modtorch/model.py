import torch, sys
from pathlib import Path
from torch import nn
from importlib import import_module

class NN_Model(nn.Module):

    def __init__(self, layers):
        super().__init__()
        if not (layers[0].get('add_input', False) or isinstance(layers[0].get('sel_input', None), int)):
            raise ValueError("First layer must be 'add_input' or 'sel_input")
        all_modules = list({d['module'] for d in layers if 'module' in d})
        if 'basic' not in all_modules: all_modules.append('basic')
        all_modules_dict = {}
        from_dir = Path(sys.argv[0]).resolve().parent
        if str(from_dir) not in sys.path:
            sys.path.insert(0, str(from_dir))
        for x in all_modules:
            try:
                all_modules_dict[x] = import_module(x)
            except ModuleNotFoundError:
                all_modules_dict[x] = import_module(f".{x}", package=__package__)
        self.layers = nn.ModuleList()
        aux_check = False
        for l in layers:
            l.setdefault('layer', 'Identity')
            dict_layer = nn.ModuleDict({'layer': _get_layer(l, all_modules_dict)})
            dict_layer.layer_name = l.get('layer_name', '')
            if 'name' in l:
                dict_layer.fn_type = 'name'
                dict_layer.name = l['name']
            elif 'name_list' in l:
                dict_layer.fn_type = 'name_list'
                dict_layer.name_list = l['name_list']
            elif l.get('add_input', False):
                if aux_check:
                    raise Exception('\nAdding an input after add_quantile_mc is not permitted.')
                dict_layer.fn_type = 'add_input'
            elif isinstance(l.get('sel_input', None), int):
                dict_layer.fn_type = 'sel_input'
                dict_layer.sel_input = l['sel_input']
            elif l.get('add_quantile_mc', False):
                aux_check = True
                dict_layer.fn_type = 'add_quantile_mc'
            elif 'output_list' in l:
                dict_layer.fn_type = 'output_list'
                dict_layer.output_list = l['output_list']
            elif 'output_dict' in l:
                dict_layer.fn_type = 'output_dict'
                dict_layer.output_dict = l['output_dict']
            else:
                dict_layer.fn_type = 'default'
            dict_layer.save = l.get('save', False)
            dict_layer.cov_subnet = l.get('cov_subnet', False)
            dict_layer.encoder = l.get('encoder', False)
            self.layers.append(dict_layer)

    @staticmethod
    def _run_layers(layers, input_list):
        i_input = 0
        save_layers = {}
        for layer_i in layers:
            if layer_i.fn_type == 'name':
                out_nn = layer_i['layer'](save_layers[layer_i.name])
            elif layer_i.fn_type == 'name_list':
                out_nn = layer_i['layer']([save_layers[x] for x in layer_i.name_list])
            elif layer_i.fn_type in ['add_input','add_quantile_mc']:
                out_nn = input_list[i_input]
                i_input += 1
            elif layer_i.fn_type == 'sel_input':
                out_nn = input_list[layer_i.sel_input]
            elif layer_i.fn_type == 'output_list':
                out_nn = [save_layers[x] for x in layer_i.output_list]
            elif layer_i.fn_type == 'output_dict':
                out_nn = {k: save_layers[v] for k, v in layer_i.output_dict.items()}
            elif layer_i.fn_type == 'default':
                out_nn = layer_i['layer'](out_nn)
            if layer_i.save:
                save_labels = layer_i.save
                if isinstance(save_labels, (list, tuple)):
                    for k, l in enumerate(save_labels):
                        save_layers[l] = out_nn[k]
                else:
                    save_layers[layer_i.save] = out_nn
        return out_nn

    def forward(self, input_list):
        return self._run_layers(self.layers, input_list)

    def encoder(self, input_list):
        enc_layers = [l for l in self.layers if l.encoder]
        return self._run_layers(enc_layers, input_list)

class MOE_Model(nn.Module):
    def __init__(self, expert_layers, gathing_layers, gathing='soft_selection'):
        super().__init__()
        if not (expert_layers[0][0].get('add_input', False) or isinstance(expert_layers[0][0].get('sel_input', None), int)):
            raise ValueError("First layer must be Identity with 'add_input' or 'sel_input.")
        all_modules = list({d['module'] for d in expert_layers+gathing_layers if 'module' in d})
        if 'basic' not in all_modules: all_modules.append('basic')
        all_modules_dict = {}
        for x in all_modules:
            all_modules_dict[x] = import_module(f".{x}", package=__package__)
        self.n_experts = len(expert_layers)
        self.expert_layers = nn.ModuleList()
        self.gathing = gathing
        for k in range(self.n_experts):
            layers_list = nn.ModuleList()
            for l in expert_layers[k]:
                dict_layer = nn.ModuleDict({'layer': _get_layer(l, all_modules_dict)})
                dict_layer.layer_name = l.get('layer_name', '')
                if 'name' in l:
                    dict_layer.fn_type = 'name'
                    dict_layer.name = l['name']
                elif 'name_list' in l:
                    dict_layer.fn_type = 'name_list'
                    dict_layer.name_list = l['name_list']
                elif l.get('add_input', False):
                    dict_layer.fn_type = 'add_input'
                elif isinstance(l.get('sel_input', None), int):
                    dict_layer.fn_type = 'sel_input'
                    dict_layer.sel_input = l['sel_input']
                elif 'output_list' in l:
                    dict_layer.fn_type = 'output_list'
                    dict_layer.output_list = l['output_list']
                else:
                    dict_layer.fn_type = 'default'
                dict_layer.save = l.get('save', False)
                dict_layer.cov_subnet = l.get('cov_subnet', False)
                layers_list.append(dict_layer)
            self.expert_layers.append(layers_list)
        layers_list = nn.ModuleList()
        for l in gathing_layers:
            dict_layer = nn.ModuleDict({'layer': _get_layer(l, all_modules_dict)})
            dict_layer.layer_name = l.get('layer_name', '')
            if 'name' in l:
                dict_layer.fn_type = 'name'
                dict_layer.name = l['name']
            elif 'name_list' in l:
                dict_layer.fn_type = 'name_list'
                dict_layer.name_list = l['name_list']
            elif l.get('add_input', False):
                dict_layer.fn_type = 'add_input'
            elif isinstance(l.get('sel_input', None), int):
                dict_layer.fn_type = 'sel_input'
                dict_layer.sel_input = l['sel_input']
            elif 'output_list' in l:
                dict_layer.fn_type = 'output_list'
                dict_layer.output_list = l['output_list']
            else:
                dict_layer.fn_type = 'default'
            dict_layer.save = l.get('save', False)
            dict_layer.cov_subnet = l.get('cov_subnet', False)
            layers_list.append(dict_layer)
        self.gathing_layers = layers_list

    def forward(self, input_list, gathing_prob=False):
        # experts net
        out_experts = torch.stack([NN_Model._run_layers(self.expert_layers[i], input_list)
            for i in range(self.n_experts)], dim=-1)
        # gathing net
        out_gathing = NN_Model._run_layers(self.gathing_layers, input_list)
        if self.gathing == 'soft_selection':
            out_nn = torch.sum(out_experts * out_gathing.unsqueeze(1), dim=-1)
        elif self.gathing == 'hard_selection':
            out_gathing_one_hot = nn.functional.one_hot(torch.argmax(out_gathing, dim=1), num_classes=self.n_experts)
            out_nn = torch.sum(out_experts * out_gathing_one_hot.unsqueeze(1), dim=-1)
        if gathing_prob:
            return out_nn, out_gathing
        else:
            return out_nn

def _get_layer(l, all_modules_dict):
    module_name = l.get('module', 'basic')
    layer_type = l.get('layer', None)
    x = getattr(all_modules_dict[module_name], layer_type)(l)
    return x
