# coding:utf-8
from __future__ import print_function
from collections import OrderedDict
import torch
from torch.autograd import Variable
import torch.nn as nn
from models import hourglass, resnet
from models.resnet_fc import fcResnet
import json


def get_output_size(summary_dict, output):
    if isinstance(output, tuple):
        for i in range(len(output)):
            summary_dict[i] = OrderedDict()
            summary_dict[i] = get_output_size(summary_dict[i], output[i])
    else:
        summary_dict['output_shape'] = list(output.size())
    return summary_dict


def summary(input_size, model):
    def register_hook(module):
        def hook(module, input, output):
            class_name = str(module.__class__).split('.')[-1].split("'")[0]
            module_idx = len(summary)

            m_key = '%s-%i' % (class_name, module_idx + 1)
            summary[m_key] = OrderedDict()
            summary[m_key]['input_shape'] = list(input[0].size())
            summary[m_key] = get_output_size(summary[m_key], output)

            params = 0
            if hasattr(module, 'weight'):
                params += torch.prod(torch.LongTensor(list(module.weight.size())))
                if module.weight.requires_grad:
                    summary[m_key]['trainable'] = True
                else:
                    summary[m_key]['trainable'] = False
            # if hasattr(module, 'bias'):
            #  params +=  torch.prod(torch.LongTensor(list(module.bias.size())))

            summary[m_key]['nb_params'] = params

        if not isinstance(module, nn.Sequential) and \
                not isinstance(module, nn.ModuleList) and \
                not (module == model):
            hooks.append(module.register_forward_hook(hook))

    # check if there are multiple inputs to the network
    if isinstance(input_size[0], (list, tuple)):
        x = [Variable(torch.rand(1, *in_size)) for in_size in input_size]
    else:
        x = Variable(torch.rand(1, *input_size))

    # create properties
    summary = OrderedDict()
    hooks = []
    # register hook
    model.apply(register_hook)
    # make a forward pass
    model(x)
    # remove these hooks
    for h in hooks:
        h.remove()

    return summary


hg_model = hourglass.hg_ver2()
res_fc_model = fcResnet(16)
res_model = resnet.resnet18()

x_res_fc = summary([3, 256, 256], res_fc_model)
x_res = summary([3, 256, 256], res_model)
x_hg = summary([3, 256, 256], hg_model)

with open('res_fc_out.json', 'w') as f:
    # print(json.dumps(x_res_fc))
    json.dump(x_res_fc, f)

with open('res_out.json', 'w') as f:
    json.dump(x_res, f)

with open('hg_out.json', 'w') as f:
    json.dump(x_hg, f)
