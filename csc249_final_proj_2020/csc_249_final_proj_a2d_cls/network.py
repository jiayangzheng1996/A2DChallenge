import torch
import torch.nn as nn
import torchvision.models as models
import torch.autograd as autograd
from torch.autograd import Variable
import math

class net(nn.Module):
    def __init__(self, args):
        super(net, self).__init__()
        resnet = models.resnet152(pretrained=True)
        modules = list(resnet.children())[:-1]
        self.res152 = nn.Sequential(*modules)
        self.fc = nn.Linear(resnet.fc.in_features, args.num_cls)
        self.bn = nn.BatchNorm1d(args.num_cls)
    def forward(self, image):
        with torch.no_grad():
            outputs = self.res152(image)
        outputs = outputs.reshape(outputs.shape[0], -1)
        outputs = self.fc(outputs)
        outputs = self.bn(outputs)
        return outputs
