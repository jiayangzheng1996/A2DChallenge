import torch
import torch.nn as nn
import torchvision.models as models
import torch.autograd as autograd
from torch.autograd import Variable
import math


class net(nn.Module):
    def __init__(self, args, mode='train'):
        super(net, self).__init__()
        self.actor = ActorClassifier(args)
        self.actor_action = ActorActionClassifier(args)
        self.mode = mode

    def forward(self, image, image_sequence):
        actor = self.actor(image)
        actor_action = self.actor_action(image_sequence)

        if self.mode == 'test':
            self.hierarchical(actor, actor_action)
            return actor_action

        return actor, actor_action

    def hierarchical(self, actor, actor_action):
        actor = torch.sigmoid(actor)
        actor_action = torch.sigmoid(actor_action)

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


class ActorClassifier(nn.Module):
    def __init__(self, args):
        super(ActorClassifier, self).__init__()
        resnet1 = models.resnet152(pretrained=True)
        modules = list(resnet1.children())[:-1]
        self.res152 = nn.Sequential(*modules)
        self.fc1 = nn.Sequential(
            nn.Linear(resnet1.fc.in_features, 7),
            nn.Dropout(p=0.5)
        )

    def forward(self, images):
        with torch.no_grad():
            features1 = self.res152(images)
        features1 = features1.reshape(features1.shape[0], -1)
        actor = self.fc1(features1)
        return actor


class Encoder(nn.Module):
    def __init__(self, embed_size):
        """Load the pretrained ResNet-152 and replace top fc layer."""
        super(Encoder, self).__init__()
        resnet = models.resnet152(pretrained=True)
        modules = list(resnet.children())[:-1]  # delete the last fc layer.
        self.resnet = nn.Sequential(*modules)
        self.linear = nn.Linear(resnet.fc.in_features, embed_size)
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)

    def forward(self, images):
        """Extract feature vectors from input images."""
        with torch.no_grad():
            features = self.resnet(images)
        features = features.reshape(features.size(0), -1)
        features = self.bn(self.linear(features))
        return features


class Decoder(nn.Module):
    def __init__(self, latent_dim, num_layers, hidden_dim, bidirectional):
        super(Decoder, self).__init__()
        self.lstm = nn.LSTM(latent_dim, hidden_dim, num_layers, batch_first=True, bidirectional=bidirectional)
        self.hidden_state = None

    def __reset__hidden__(self):
        self.hidden_state = None

    def forward(self, features):
        output, self.hidden_state = self.lstm(features, self.hidden_state)
        return output


class ActorActionClassifier(nn.Module):
    def __init__(self, args, latent_dim=512, num_layers=1, hidden_dim=1024, bidirectional=True):
        super(ActorActionClassifier, self).__init__()
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim, num_layers, hidden_dim, bidirectional=bidirectional)
        self.fc = nn.Sequential(
            nn.Linear(2 * hidden_dim if bidirectional == True else hidden_dim, hidden_dim),
            nn.Dropout(p=0.5),
            nn.ReLU(),
            nn.Linear(hidden_dim, args.num_cls)
        )

    def forward(self, image_sequence):
        num_batch, num_sequence, c, h, w = image_sequence.shape
        images = image_sequence.reshape(num_batch * num_sequence, c, h, w)
        features = self.encoder(images)
        features = features.reshape(num_batch, num_sequence, -1)
        output = self.decoder(features)
        output = self.fc(output)
        output = output.mean(1)
        return output
