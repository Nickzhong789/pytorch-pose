from pose.models import resnet
from pose.models import hourglass
from pose.models.resnet_fc import fcResnet
from torch.autograd import Variable
import torch


def get_features_hook(output):
    print("hook", output.data.numpy().shape)


x = Variable(torch.rand([6, 3, 256, 256]))
res_model = resnet.resnet18()
hg_model = hourglass.hg_ver2()
res_fc_model = fcResnet(16)

model = raw_input("Please select model: ")
if model == 'resnet':
    print(res_model)
elif model == 'hg':
    print(hg_model)
elif model == 'resnet_fc':
    y = res_fc_model(x)
    handle = res_fc_model.classifier[-2].register_forward_hook(get_features_hook)
    print(handle)
    handle.remove()
