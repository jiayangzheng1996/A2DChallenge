from loader import a2d_dataset
import argparse
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0" # GPU ID
from torch.utils.data import Dataset, DataLoader
from cfg.deeplab_pretrain_a2d import train as train_cfg
from cfg.deeplab_pretrain_a2d import val as val_cfg
from cfg.deeplab_pretrain_a2d import test as test_cfg
from network import net
from utils.eval_metrics import Precision, Recall, F1
import time

# use gpu if cuda can be detected
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_actor_cls(labels):
    num_batch, num_cls = labels.shape
    valid_cls = a2d_dataset.A2DDataset.valid_cls

    actor_labels = torch.zeros([num_batch, 7])

    for b in range(num_batch):
        for c in range(num_cls):
            if labels[b, c].item():
                cls = valid_cls[c]
                cls_str = str(cls)
                actor = int(cls_str[0]) - 1
                actor_labels[b, actor] = 1

    return actor_labels
    

def validate(model, args, epoch, f):
    model.mode = 'test'

    test_dataset = a2d_dataset.A2DDatasetVideo(val_cfg, args.dataset_path)
    data_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1)

    X = np.zeros((data_loader.__len__(), args.num_cls))
    Y = np.zeros((data_loader.__len__(), args.num_cls))
    print(data_loader.__len__())
    model.eval()
    with torch.no_grad():
        for batch_idx, data in enumerate(data_loader):
            # mini-batch
            image = data[0].to(device)
            images = data[1].to(device)
            labels = data[2].type(torch.FloatTensor).to(device)
            model.actor_action.decoder.__reset__hidden__()
            output = model(image, images).cpu().detach().numpy()
            target = labels.cpu().detach().numpy()
            output[output >= 0.5] = 1
            output[output < 0.5] = 0
            X[batch_idx, :] = output
            Y[batch_idx, :] = target

    P = Precision(X, Y)
    R = Recall(X, Y)
    F = F1(X, Y)
    print('Precision: {:.1f} Recall: {:.1f} F1: {:.1f}'.format(100 * P, 100 * R, 100 * F))

    f.write("Epoch {}: Precision: {:.1f} Recall: {:.1f} F1: {:.1f} \n".format(epoch, 100 * P, 100 * R, 100 * F))
    f.flush()

    model.mode = 'train'
    return P+R+F


def main(args):
    # Create model directory for saving trained models
    f = open("result_hierarchical.txt", "w")
    f.write("Following are the validation result: \n")
    f.flush()

    best_metrics = 0

    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)

    test_dataset = a2d_dataset.A2DDatasetVideo(train_cfg, args.dataset_path)
    data_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers) # you can make changes

    # Classifier config
    model = net(args).to(device)###
    criterion_actor = nn.BCEWithLogitsLoss()
    criterion_actor_action = nn.BCEWithLogitsLoss()###
    # params = list(model.fc1.parameters()) + list(model.fc2.parameters()) + list(model.spatial_conv.parameters()) + \
    #          list(model.spatial_fc.parameters()) + list(model.weight_net.parameters())
    params = model.parameters()

    #net config
    # model = net(args).to(device)
    # criterion = nn.BCEWithLogitsLoss()
    # params = list(model.fc.parameters())

    optimizer = optim.Adam(params, lr=args.lr)###

    lr_decay = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=args.gamma)

    best_metrics = validate(model, args, 0, f)

    # Train the models
    total_step = len(data_loader)
    for epoch in range(1, args.num_epochs+1):
        model.train()
        t1 = time.time()
        for i, data in enumerate(data_loader):

            # mini-batch
            # image = Variable(data[0].to(device), requires_grad=True)
            # images = Variable(data[1].to(device), requires_grad=True)
            # labels = Variable(data[2].type(torch.FloatTensor).to(device), requires_grad=True)
            image = data[0].to(device)
            images = data[1].to(device)
            labels = data[2].type(torch.FloatTensor).to(device)
            actor_labels = get_actor_cls(labels).to(device)

            optimizer.zero_grad()

            model.actor_action.decoder.__reset__hidden__()

            # Forward, backward and optimize
            actor, actor_action = model(image, images)
            loss = criterion_actor(actor, actor_labels) + criterion_actor_action(actor_action, labels)
            loss = loss.mean()
            loss.backward()
            optimizer.step()

            # Log info
            if i % args.log_step == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                      .format(epoch, args.num_epochs, i, total_step, loss.item()))

            # Save the model checkpoints
            # if (i + 1) % args.save_step == 0:
            #     torch.save(model.state_dict(), os.path.join(
            #         args.model_path, 'net.ckpt'))
        t2 = time.time()
        print(t2 - t1)
        if epoch % args.log_step == 0 or epoch % args.log_step == 5:
            metrics = validate(model, args, epoch, f)
            if metrics > best_metrics:
                torch.save(model.state_dict(), os.path.join(args.model_path, 'net.ckpt'))
                best_metrics = metrics

        lr_decay.step()

    f.close()
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='models/', help='path for saving trained models')
    parser.add_argument('--dataset_path', type=str, default='../A2D', help='a2d dataset')
    parser.add_argument('--log_step', type=int, default=10, help='step size for prining log info')
    parser.add_argument('--save_step', type=int, default=1000, help='step size for saving trained models')
    parser.add_argument('--num_cls', type=int, default=43)
    # Model parameters
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--gamma', type=float, default=0.95)
    args = parser.parse_args()
    print(args)
main(args)
