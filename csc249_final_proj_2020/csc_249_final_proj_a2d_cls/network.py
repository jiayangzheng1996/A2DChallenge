import torch
import torch.nn as nn
import torchvision.models as models
import torch.autograd as autograd
from torch.autograd import Variable
import math


def outputs_vectorize(output1, output2, output3):
    num_batch, num_cls = output1.shape

    list = []

    for i in range(num_cls):
        vector = torch.zeros([num_batch, 3])
        for b in range(num_batch):
            vector[b, 0] = output1[b, i]
            vector[b, 1] = output2[b, i]
            vector[b, 2] = output3[b, i]
        list.append(vector)

    return list


class net(nn.Module):
    def __init__(self, args):
        super(net, self).__init__()
        resnet = models.resnet152(pretrained=True)
        modules = list(resnet.children())[:-1]
        self.res152 = nn.Sequential(*modules)
        self.fc = nn.Sequential(
            nn.Linear(resnet.fc.in_features, args.num_cls),
            nn.BatchNorm1d(args.num_cls, momentum=0.01)
        )

    def forward(self, image):
        with torch.no_grad():
            outputs = self.res152(image)
        outputs = outputs.reshape(outputs.shape[0], -1)
        outputs = self.fc(outputs)
        # outputs = self.bn(outputs)
        return outputs


class Classifier(nn.Module):
    def __init__(self, args, mode='train'):
        super(Classifier, self).__init__()

        self.mode = mode

        resnet1 = models.resnet152(pretrained=True)
        modules = list(resnet1.children())[:-1]
        self.res152 = nn.Sequential(*modules)
        self.fc1 = nn.Sequential(
            nn.Linear(resnet1.fc.in_features, 2048),
            nn.Dropout(p=0.5),
            nn.BatchNorm1d(2048, momentum=0.01)
        )
        resnet2 = models.resnet101(pretrained=True)
        modules = list(resnet2.children())[:-1]
        self.res101 = nn.Sequential(*modules)
        self.fc2 = nn.Sequential(
            nn.Linear(resnet2.fc.in_features, 2048),
            nn.Dropout(p=0.5),
            nn.BatchNorm1d(2048, momentum=0.01)
        )

        self.spatial_conv = nn.Sequential(
            nn.Conv2d(3, 96, 7, stride=2),
            nn.BatchNorm2d(96),
            nn.AvgPool2d(2),

            nn.Conv2d(96, 256, 5, stride=2),
            nn.BatchNorm2d(256),
            nn.AvgPool2d(2),

            nn.Conv2d(256, 512, 3),

            nn.Conv2d(512, 512, 3),

            nn.Conv2d(512, 512, 3),
            nn.AvgPool2d(2)
        )
        self.spatial_fc = nn.Sequential(
            nn.Linear(4608, 2048),
            nn.Dropout(p=0.5),
            nn.BatchNorm1d(2048, momentum=0.01)
        )

        self.fc_cat = nn.Sequential(
            nn.Linear(3 * 2048, 2048),
            nn.Dropout(p=0.5),
            nn.Sigmoid()
        )

        self.fc_actor = nn.Sequential(
            nn.Linear(2048, 7),
            nn.Dropout(p=0.5),
            nn.Sigmoid()
        )

        self.fc_actor_action = nn.Sequential(
            nn.Linear(2048, args.num_cls),
            nn.Dropout(p=0.5),
            nn.Sigmoid()
        )

    def forward(self, images):
        with torch.no_grad():
            features1 = self.res152(images)
            features2 = self.res101(images)
        features1 = features1.reshape(features1.shape[0], -1)
        features2 = features2.reshape(features2.shape[0], -1)
        output1 = self.fc1(features1)
        output2 = self.fc2(features2)
        # output1 = features1
        # output2 = features2


        features3 = self.spatial_conv(images)
        features3 = features3.reshape(features3.shape[0], -1)
        output3 = self.spatial_fc(features3)

        outputs = torch.cat((output1, output2, output3), -1)
        outputs = self.fc_cat(outputs)

        actor = self.fc_actor(outputs)
        actor_action = self.fc_actor_action(outputs)

        if self.mode == 'test':
            self.hierarchical(actor, actor_action)
            return actor_action

        return actor, actor_action

    def hierarchical(self, actor, actor_action):
        num_batch, num_cls = actor_action.shape

        if self.mode == 'test':
            for b in range(num_batch):
                for c in range(num_cls):
                    if 0 <= c < 8:
                        actor_action[b, c].data *= actor[b, 0].data
                    elif 8 <= c < 13:
                        actor_action[b, c].data *= actor[b, 1].data
                    elif 13 <= c < 17:
                        actor_action[b, c].data *= actor[b, 2].data
                    elif 17 <= c < 24:
                        actor_action[b, c].data *= actor[b, 3].data
                    elif 24 <= c < 29:
                        actor_action[b, c].data *= actor[b, 4].data
                    elif 29 <= c < 36:
                        actor_action[b, c].data *= actor[b, 5].data
                    else:
                        actor_action[b, c].data *= actor[b, 6].data
        else:
            print("mode is not testing when reached hierarchical joint prediction. \n")