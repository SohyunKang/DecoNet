# Copyright (c) 2024, Sohyun Kang
# All rights reserved.

# This code is part of the research work presented in:
# "Disentangling brain atrophy heterogeneity in Alzheimer’s disease:
# a deep self-supervised approach with interpretable latent space" by Sohyun Kang, published in Neuroimage, 2024.


# This code is based on the implementation of deepcluster by facebookresearch,
# available at https://github.com/facebookresearch/deepcluster.

import argparse
import time
from datetime import datetime
import random
import os

from src import network
import clustering
import corticaldataset
from src.util import label_stab, UnifLabelSampler, AverageMeter, plotandsave, load_model, load_gmm

from sklearn.metrics import adjusted_mutual_info_score
from sklearn.model_selection import KFold
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data



def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch Implementation of DebaNet')
    parser.add_argument('--mode', type=str, choices=['train', 'test'], default='train',
                        help='Mode setting: train or test (default: train)')
    parser.add_argument('--data', type=str, choices=['ADNI', 'ADNI_mci', 'ADNI_long', 'OASIS'],
                        default='ADNI', help='Data type to use (default: ADNI)')
    parser.add_argument('--num_cluster', '--k', type=int, default=4,
                        help='Number of clusters for k-means and Gaussian mixture model (default: 4)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate (default: 0.001)')
    parser.add_argument('--reassign', type=float, default=1.0,
                        help='Epochs between consecutive reassignments of clusters (default: 1)')
    parser.add_argument('--wd', type=float, default=-5,
                        help='Weight decay power (default: -5)')
    parser.add_argument('--workers', type=int, default=4,
                        help='Number of data loading workers (default: 4)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Total number of epochs to run (default: 100)')
    parser.add_argument('--batch', type=int, default=64,
                        help='Mini-batch size (default: 64)')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='Momentum (default: 0.9)')
    parser.add_argument('--seed', type=int, default=31,
                        help='Random seed (default: 31)')
    parser.add_argument('--fold_seed', type=int, default=31,
                        help='Random seed for cross-validation (default: 31)')
    parser.add_argument('--fold_idx', type=int, default=0,
                        help='Fold index for cross-validation')
    parser.add_argument('--model', type=str, default='', metavar='PATH',
                        help='Path to the model (default: None)')
    parser.add_argument('--gmmmodel', type=str, default='', metavar='PATH',
                        help='Path to the clustering model (default: None)')

    return parser.parse_args()

def compute_features(dataloader, model, N):
    model.eval()
    for i, (input_tensor) in enumerate(dataloader):
        input_var = input_tensor.to(torch.device('cuda'))
        aux = model(input_var).data.cpu().numpy()
        if i == 0:
            features = np.zeros((N, aux.shape[1]), dtype='float32')
        aux = aux.astype('float32')
        if i < len(dataloader) - 1:
            features[i * args.batch: (i + 1) * args.batch] = aux
        else:
            # special treatment for final batch
            features[i * args.batch:] = aux

    return features

def train(loader, model, crit, opt):
    """Training of the encoder and classifier.
        Args:
            loader (torch.utils.data.DataLoader): Data loader
            model (nn.Module): mlp
            crit (torch.nn): loss
            opt (torch.optim.SGD): optimizer for every parameters with True
                                   requires_grad in model except top layer
    """
    model.train()
    # create an optimizer for the last fc layer
    optimizer_tl = torch.optim.SGD(
        model.top_layer.parameters(),
        lr=args.lr,
        weight_decay=10**args.wd,
    )

    for i, (input_tensor, target) in enumerate(loader):
        target = target.cuda()
        input_var = input_tensor.to(torch.device('cuda'))
        target_var = target.to(torch.device('cuda'))
        output = model(input_var)
        loss = crit(output, target_var)

        # compute gradient and do SGD step
        opt.zero_grad()
        optimizer_tl.zero_grad()
        loss.backward()
        opt.step()
        optimizer_tl.step()

        torch.cuda.memory_allocated()

def feedforward(loader, model, crit):
    """Just feedforward.
        Args:
            loader (torch.utils.data.DataLoader): Data loader
            model (nn.Module): mlp
            crit (torch.nn): loss
    """
    losses = AverageMeter()
    # switch to train mode
    model.eval()
    total = 0
    correct = 0
    for i, (input_tensor, target) in enumerate(loader):
        target = target.cuda()
        input_var = torch.autograd.Variable(input_tensor.cuda())
        target_var = torch.autograd.Variable(target)
        output = model(input_var)
        if i == 0:
            outputs = output
        else:
            outputs = torch.cat((outputs, output), dim=0)
        loss = crit(output, target_var)
        _, predicted = torch.max(output, 1)
        total += target.size(0)
        correct += (predicted == target_var).sum().item()
        # record loss
        losses.update(loss.item(), input_tensor.size(0))
        torch.cuda.memory_allocated()

    return losses.avg,  100*correct/total, outputs.cpu().detach().numpy()

args = parse_args()
if args.mode == 'train':
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        print("/// Cuda is available ///")
        torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    print('K : {}'.format(args.num_cluster))
    print('Data: {}'.format(args.data))
    print('learning rate: {}'.format(args.lr))
    print('reassign : {}'.format(args.reassign))

    model = network.mlp(input_dim=1193, output_dim=args.num_cluster)
    fd = int(model.top_layer.weight.size()[1])
    model.top_layer = None
    model.cuda()
    cudnn.benchmark = True

    # optimizer
    optimizer = torch.optim.SGD(
        filter(lambda x: x.requires_grad, model.parameters()),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=10**args.wd,
    )

    # loss function
    criterion = nn.CrossEntropyLoss().cuda()

    # load data
    end = time.time()
    print("loading the dataset...")
    dataset = corticaldataset.CorticalDataset(datatype=args.data)
    print('Data loading time: {0:.2f} s'.format(time.time() - end))
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=args.batch,
                                             num_workers=args.workers,
                                             pin_memory=True, )

    # cross-validation
    assert 0 <= args.fold_idx and args.fold_idx < 10
    skf = KFold(n_splits=10, shuffle=True, random_state=args.fold_seed)
    idx_list = []
    for idx in skf.split(np.zeros(len(dataset))):
        idx_list.append(idx)
    train_index, test_index = idx_list[args.fold_idx]

    # Gaussian mixture model
    GMM = clustering.GMM(args.num_cluster)

    # log
    results = {
        'train_loss': [],
        'train_accuracy': [],
        'train_ami': [],
        'val_loss': [],
        'val_accuracy': [],
        'val_ami': [],
        'features': {},    # epoch x subject x latent dim
        'labels': {},      # epoch x subject x 1
        'val_predictions': {},  # epoch x subject x clusters
    }

    for epoch in range(args.epochs):
        end = time.time()
        model.top_layer = None
        model.features = nn.Sequential(*list(model.features.children())[:-1]) # what for?
        features = compute_features(dataloader, model, len(dataset))    # NUMPY format

        # cluster the features
        label_unmatched, gmm_params = GMM.cluster(features, train_index, test_index)
        images_lists_unmatched = GMM.images_lists

        if epoch == 0:
            label_prev = label_unmatched.astype(int)
            images_lists_prev = images_lists_unmatched
        else:
            label_prev = label_now
            images_lists_prev = images_lists_now

        # label stabilization for calculating accuracy
        label_now, images_lists_now = label_stab(images_lists_unmatched, images_lists_prev, label_unmatched)

        results['features'][f'epoch_{epoch + 1}'] = features.tolist()
        results['labels'][f'epoch_{epoch+1}'] = label_now.tolist()

        train_ami = adjusted_mutual_info_score(label_prev[train_index], label_now[train_index])
        val_ami = adjusted_mutual_info_score(label_prev[test_index], label_now[test_index])
        results['train_ami'].append(train_ami)
        results['val_ami'].append(val_ami)

        # assign pseudo-labels
        train_dataset = clustering.ReassignedDataset(dataset.x_data.numpy()[train_index],
                                                     label_now[train_index])
        test_dataset = clustering.ReassignedDataset(dataset.x_data.numpy()[test_index],
                                                    label_now[test_index])

        # sampler
        images_lists_train_abs = [[] for i in range(len(images_lists_now))]
        images_lists_test_abs = [[] for i in range(len(images_lists_now))]
        train_index = np.array(list(set(train_index)), dtype=int)
        test_index = np.array(list(set(test_index)), dtype=int)
        for label, image_list in enumerate(images_lists_now):
            images_lists_train_abs[label] = [i for i, idx in enumerate(train_index) if idx in image_list]
            images_lists_test_abs[label] = [i for i, idx in enumerate(test_index) if idx in image_list]
        train_sampler = UnifLabelSampler(int(args.reassign * len(train_dataset)), images_lists_train_abs)
        test_sampler = UnifLabelSampler(int(args.reassign * len(test_dataset)), images_lists_test_abs)

        # Load pseudo-labeled train and test datasets
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batch,
            num_workers=args.workers,
            sampler=train_sampler,
            pin_memory=True)

        test_dataloader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=args.batch,
            num_workers=args.workers,
            pin_memory=True)

        # set top_layer
        tmp = list(model.features.children())
        tmp.append(nn.ReLU().cuda())
        model.features = nn.Sequential(*tmp)
        if epoch == 0:
            model.top_layer = nn.Linear(fd, len(images_lists_now))
            model.top_layer.weight.data.normal_(0, 0.01)
            model.top_layer.bias.data.zero_()
            model.top_layer.cuda()
        else:
            model.top_layer = top_layer_tmp
            model.top_layer.cuda()

        loss_t = train(train_dataloader, model, criterion, optimizer)
        top_layer_tmp = model.top_layer
        train_dataloader_wo_sampling = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batch,
            num_workers=args.workers,
            pin_memory=True)

        loss_t, acc_t, output_train = feedforward(train_dataloader_wo_sampling, model, criterion)
        loss_v, acc_v, output_test = feedforward(test_dataloader, model, criterion)

        results['train_loss'].append(loss_t)
        results['train_accuracy'].append(acc_t)
        results['val_loss'].append(loss_v)
        results['val_accuracy'].append(acc_v)
        results['val_predictions'][f'epoch_{epoch + 1}'] = output_test.tolist()

        print(
            'Epoch [{0}]| Total time: {1:.2f} s | Train loss: {2:.2f} | Val loss: {3:.2f} | Train acc: {4: .1f} | Val acc: {5: .1f} | Train AMI: {6: .1f} | Val AMI: {7: .1f} | Inclass size: '
            .format(epoch, time.time() - end,  loss_t, loss_v,
                    acc_t, acc_v, train_ami, val_ami), [len(x) for x in images_lists_now])

    script_dir = os.path.dirname(__file__)
    now = datetime.now()
    datetime_string = now.strftime("%Y%m%d%H%M%S")
    profile = '_%s_%s_%s_k=%d_lr=%f_reassign=%d_ep=%d_fold_idx=%d' \
              % (datetime_string, args.mode, args.data, args.num_cluster, args.lr, args.reassign, args.epochs,
                 args.fold_idx)
    results_dir = os.path.join(script_dir, 'Results' + profile)
    plotandsave(results_dir, results, train_index, test_index, gmm_params, model, optimizer, epoch)


elif args.mode == 'test':
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        print("/// Cuda is available ///")
        torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # load model and GMM
    model = load_model(args.model)
    gmm = load_gmm(args.gmmmodel)
    model.cuda()
    cudnn.benchmark = True

    # load data
    end = time.time()
    print("loading the dataset...")
    dataset = corticaldataset.CorticalDataset(mode=args.mode)
    print(dataset, len(dataset))
    print('Data loading time: {0:.2f} s'.format(time.time() - end))

    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=args.batch,
                                             num_workers=args.workers,
                                             pin_memory=True, )

    # compute features and pseudo-labels
    top_layer = model.top_layer
    model.top_layer = None
    model.features = nn.Sequential(*list(model.features.children())[:-1])
    feature = compute_features(dataloader, model, len(dataset))
    label = gmm.predict(feature)

    # add classifier
    tmp = list(model.features.children())
    tmp.append(nn.ReLU().cuda())
    model.features = nn.Sequential(*tmp)
    model.top_layer = top_layer
    model.top_layer.cuda()

    # test dataset
    test_dataset = clustering.ReassignedDataset(dataset.x_data.numpy(), label)
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch,
        num_workers=args.workers,
        pin_memory=True)

    # loss function
    criterion = nn.CrossEntropyLoss().cuda()

    # prediction
    _, _, outputs = feedforward(test_dataloader, model, criterion)
    predicted_labels = np.argmax(outputs, 1)

    # re-label
    uniques, counts = np.unique(predicted_labels, return_counts=True)
    unique_counts = list(zip(uniques, counts))

    image_list_unmatched = [[] for i in range(len(uniques))]
    for i in range(len(label)):
        image_list_unmatched[int(label[i])].append(i)
    image_list_predicted = [[] for i in range(len(uniques))]
    for i in range(len(predicted_labels)):
        image_list_predicted[int(predicted_labels[i])].append(i)
    label, _ = label_stab(image_list_unmatched, image_list_predicted, label)

    test_dataset2 = clustering.ReassignedDataset(dataset.x_data.numpy(), label)
    test_dataloader2 = torch.utils.data.DataLoader(
        test_dataset2,
        batch_size=args.batch,
        num_workers=args.workers,
        pin_memory=True)

    # compute loss and accuracy
    loss, acc, outputs = feedforward(test_dataloader2, model, criterion)

    print('Loss: {0:.2f} | Accuracy: {1:.1f} | label: '.format(loss, acc), unique_counts)

    # save outputs and labels
    script_dir = os.path.dirname(__file__)
    now = datetime.now()
    datetime_string = now.strftime("%Y%m%d%H%M%S")
    profile = '_%s_%s_%s_k=%d_lr=%f_reassign=%d_ep=%d_fold_idx=%d' \
              % (datetime_string, args.mode, args.data, args.num_cluster, args.lr, args.reassign, args.epochs,
                 args.fold_idx)
    results_dir = os.path.join(script_dir, 'Results' + profile)

    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)
    (pd.DataFrame(np.array(outputs))).to_csv(results_dir + '/' + 'outputs.csv')
    (pd.DataFrame(np.array(predicted_labels))).to_csv(results_dir + '/' + 'predicted_labels.csv')



