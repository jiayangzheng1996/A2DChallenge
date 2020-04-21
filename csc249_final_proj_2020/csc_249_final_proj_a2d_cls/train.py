from loader import a2d_dataset
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0" # GPU ID
from torch.utils.data import Dataset, DataLoader
from cfg.deeplab_pretrain_a2d import train as train_cfg
from cfg.deeplab_pretrain_a2d import val as val_cfg
from cfg.deeplab_pretrain_a2d import test as test_cfg
from network import Classifier, net
from utils.eval_metrics import Precision, Recall, F1
import time

# use gpu if cuda can be detected
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def validate(model, args, epoch, f):
    test_dataset = a2d_dataset.A2DDataset(val_cfg, args.dataset_path)
    data_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1)

    X = np.zeros((data_loader.__len__(), args.num_cls))
    Y = np.zeros((data_loader.__len__(), args.num_cls))
    print(data_loader.__len__())
    model.eval()
    with torch.no_grad():
        for batch_idx, data in enumerate(data_loader):
            # mini-batch
            images = data[0].to(device)
            labels = data[1].type(torch.FloatTensor).to(device)
            output = model(images).cpu().detach().numpy()
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

    return P+R+F



def main(args):
    # Create model directory for saving trained models
    f = open("result.txt", "w")
    f.write("Following are the validation result: \n")
    f.flush()

    best_metrics = 0

    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)

    test_dataset = a2d_dataset.A2DDataset(train_cfg, args.dataset_path)
    data_loader = DataLoader(test_dataset, batch_size=4, shuffle=True, num_workers=args.num_workers) # you can make changes

    # Classifier config
    model = Classifier(args).to(device)###
    criterion = nn.BCELoss()###
    params = list(model.fc1.parameters()) + list(model.fc2.parameters()) + list(model.spatial_conv.parameters()) + \
             list(model.spatial_fc.parameters()) + list(model.weight_net.parameters())

    #net config
    # model = net(args).to(device)
    # criterion = nn.BCEWithLogitsLoss()
    # params = list(model.fc.parameters())

    optimizer = optim.Adam(params, lr=0.01)###

    lr_decay = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.95)

    # Train the models
    total_step = len(data_loader)
    for epoch in range(args.num_epochs):
        t1 = time.time()
        for i, data in enumerate(data_loader):

            # mini-batch
            images = data[0].to(device)
            labels = data[1].type(torch.FloatTensor).to(device)

            # Forward, backward and optimize
            outputs = model(images)
            loss = criterion(outputs, labels)
            model.zero_grad()
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
        metrics = validate(model, args, epoch, f)

        if metrics > best_metrics:
            torch.save(model.state_dict, os.path.join(args.model_path, 'net.ckpt'))
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
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=2)
    args = parser.parse_args()
    print(args)
main(args)
