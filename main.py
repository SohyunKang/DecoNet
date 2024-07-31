# Copyright (c) 2024, Sohyun Kang
# All rights reserved.

# This code is part of the research work presented in:
# "Disentangling brain atrophy heterogeneity in Alzheimer’s disease:
# a deep self-supervised approach with interpretable latent space" by Sohyun Kang, published in Neuroimage, 2024.


# This code is based on the implementation of deepcluster by facebookresearch,
# available at https://github.com/facebookresearch/deepcluster.

import argparse
import time
import network
import clustering
from util import label_stab, UnifLabelSampler, load_model
from itertools import permutations

import scipy.io
from sklearn.metrics import normalized_mutual_info_score
from sklearn.metrics import adjusted_mutual_info_score
from sklearn.manifold import TSNE
from sklearn.model_selection import KFold

import numpy as np
import pandas as pd
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import corticaldataset
import csv



import matplotlib.pyplot as plt
import os

import copy
import scipy.io as sio

def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch Implemenation of Subtyping with DeepCluster')

    parser.add_argument('--mode', type=str,
                        choices=['train', 'flatcv', 'nestedcv', 'test'], default='train',
                        help='Setting the mode (default: train)')
    parser.add_argument('--data', type=str, choices=['ADNI', 'ADNI_mci', 'ADNI_long', 'OASIS'],
                        default='ADNI', help='choose which data type you will use')
    parser.add_argument('--num_cluster', '--k', type=int, default=4,
                        help='number of cluster for k-means AND gaussian mixture model (default: 3)')
    parser.add_argument('--lr', default=0.001, type=float,
                        help='learning rate (default: 0.001)')
    parser.add_argument('--wd', default=-5, type=float,
                        help='weight decay pow (default: -5)')
    parser.add_argument('--reassign', type=float, default=1.,
                        help="""how many epochs of training between two consecutive
                        reassignments of clusters (default: 1)""")
    parser.add_argument('--workers', default=4, type=int,
                        help='number of data loading workers (default: 4')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of total epochs to run (default: 100)')
    parser.add_argument('--batch', default=64, type=int,
                        help='mini-batch size (default: 64)')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum (default: 0.9)')
    parser.add_argument('--seed', type=int, default=31, help='random seed (default: 31)')
    parser.add_argument('--fold_seed', type=int, default=31, help='random seed (default: 31)')
    parser.add_argument('--fold_idx', type=int, default=0, help='choose the fold index of cross-validation')
    parser.add_argument('--model', default='', type=str, metavar='PATH',
                        help='path to model (default: None)')
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
    """Training of the CNN.
        Args:
            loader (torch.utils.data.DataLoader): Data loader
            model (nn.Module): mlp
            crit (torch.nn): loss
            opt (torch.optim.SGD): optimizer for every parameters with True
                                   requires_grad in model except top layer
            epoch (int)
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

def feedforward(loader, model, crit=None):
    """Training of the CNN.
        Args:
            loader (torch.utils.data.DataLoader): Data loader
            model (nn.Module): CNN
            crit (torch.nn): loss
            opt (torch.optim.SGD): optimizer for every parameters with True
                                   requires_grad in model except top layer
            epoch (int)
    """
    batch_time = AverageMeter()
    losses = AverageMeter()
    data_time = AverageMeter()
    forward_time = AverageMeter()
    backward_time = AverageMeter()


    # switch to train mode
    model.eval()


    end = time.time()
    total = 0
    correct = 0
    for i, (input_tensor, target) in enumerate(loader):
        data_time.update(time.time() - end)

        target = target.cuda()
        input_var = torch.autograd.Variable(input_tensor.cuda())
        target_var = torch.autograd.Variable(target)
        output = model(input_var)
        if i == 0:
            outputs = output
            targets = target
        else:
            outputs = torch.cat((outputs, output), dim=0)
            targets = torch.cat((targets, target), dim=0)

        loss = crit(output, target_var)

        _, predicted = torch.max(output, 1)
        total += target.size(0)
        correct += (predicted == target_var).sum().item()
        # record loss
        losses.update(loss.item(), input_tensor.size(0))


        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        torch.cuda.memory_allocated()

    return losses.avg,  100*correct/total, outputs, targets


def test_(loader, model):
    """Training of the CNN.
        Args:
            loader (torch.utils.data.DataLoader): Data loader
            model (nn.Module): CNN
            crit (torch.nn): loss
            opt (torch.optim.SGD): optimizer for every parameters with True
                                   requires_grad in model except top layer
            epoch (int)
    """
    batch_time = AverageMeter()
    data_time = AverageMeter()

    # switch to train mode
    model.eval()

    end = time.time()
    total = 0
    correct = 0
    predicts = []
    for i, (input_tensor, target) in enumerate(loader):
        data_time.update(time.time() - end)

        target = target.cuda()
        input_var = torch.autograd.Variable(input_tensor.cuda())
        target_var = torch.autograd.Variable(target)
        output = model(input_var)

        _, predicted = torch.max(output, 1)
        # print('####', predicted, predicted.size)
        # print('#####', target_var, target_var.size)

        total += target.size(0)
        correct += (predicted == target_var).sum().item()
        predicts.extend(list(np.array(predicted.cpu())))
        # record loss
        # print('loss: ', loss.item(), input_tensor.size(0))



        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # if (i % 1) == 0:
        #     print('Epoch: [{0}][{1}/{2}]\t'
        #             'Time: {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
        #             'Data: {data_time.val:.3f} ({data_time.avg:.3f})\t'
        #             'Loss: {loss.val:.4f} ({loss.avg:.4f})'
        #             .format(epoch, i, len(loader), batch_time=batch_time,
        #                     data_time=data_time, loss=losses))
        torch.cuda.memory_allocated()

    print('TEST Acc: %d/%d = %d %%' % ( correct, total, 100 * correct / total ))

    return 100*correct/total, predicts


    # fix random seeds


args = parse_args()
if args.mode == 'flatcv':
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

    # train
    results = {
        'train_loss': [],
        'train_accuracy': [],
        'train_ami': [],
        'val_loss': [],
        'val_accuracy': [],
        'val_ami': [],
        'features': {},    # epoch x subject x latent dim
        'labels': {},      # epoch x subject x 1
        'predictions': {}, # epoch x subject x clusters
    }

    script_dir = os.path.dirname(__file__)
    profile = '_%s_k=%d_lr=%f_ep=%d_reassign=%d_fold_idx=%d'\
                  % (args.data, args.num_cluster, args.lr, args.epochs, args.reassign,  args.fold_idx)
    results_dir = os.path.join(script_dir, 'Results'+profile)

    for epoch in range(args.epochs):
        end = time.time()
        model.top_layer = None
        model.features = nn.Sequential(*list(model.features.children())[:-1]) # what for?
        features = compute_features(dataloader, model, len(dataset))    # NUMPY format

        # cluster the features
        label_unmatched, gmm_params = GMM.cluster(features, train_index, test_index)
        images_lists_unmatched = GMM.images_lists

        if epoch == 0:
            label_prev = label_unmatched
            images_lists_prev = images_lists_unmatched
        else:
            label_prev = label_now
            images_lists_prev = images_lists_now

        # label stabilization for calculating accuracy
        label_now, images_lists_now = label_stab(images_lists_unmatched, images_lists_prev, label_unmatched)

        results['features'][f'epoch_{epoch + 1}'].pseudolabels.append(features)
        results['labels'][f'epoch_{epoch+1}'].pseudolabels.append(label_now)

        # assign pseudo-labels
        t_2_images_lists_train = copy.deepcopy(images_lists_now)
        t_2_images_lists_test = [[] for i in range(len(t_2_images_lists_train))]
        t_2_images_lists_test_comp = [[] for i in range(len(t_2_images_lists_train))]
        t_2_images_lists_train_comp = [[] for i in range(len(t_2_images_lists_train))]
        """Segregation of
        train and test dataset"""
        for i, idx in enumerate(test_index):
            for label in range(len(t_2_images_lists_train)):
                if idx in t_2_images_lists_train[label]:
                    t_2_images_lists_train[label].remove(idx)
                    t_2_images_lists_test[label].append(idx)
                    t_2_images_lists_test_comp[label].append(i)
        for i, idx in enumerate(train_index):
            for label in range(len(t_2_images_lists_train)):
                if idx in t_2_images_lists_train[label]:
                    t_2_images_lists_train_comp[label].append(i)


        # print(t_2_images_lists_train, len(t_2_images_lists_train))
        # print(t_2_images_lists_test, len(t_2_images_lists_test))
        # print(t_2_images_lists_train_comp, len(t_2_images_lists_train_comp))
        # print(t_2_images_lists_test_comp, len(t_2_images_lists_test_comp))

        train_dataset = clustering.cluster_assign(t_2_images_lists_train_comp,
                                                 dataset.x_data.numpy()[train_index])
        test_dataset = clustering.cluster_assign(t_2_images_lists_test_comp,
                                                 dataset.x_data.numpy()[test_index])
        # print(train_dataset, len(train_dataset))

        # sampler

        train_sampler = UnifLabelSampler(int(args.reassign*len(train_dataset)), t_2_images_lists_train_comp)
        test_sampler = UnifLabelSampler(int(args.reassign * len(test_dataset)), t_2_images_lists_test_comp)

        # Load pseudo-labeled datasets
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
            # sampler=test_sampler,
            pin_memory=True)

        # set top_layer
        if not args.model:
            tmp = list(model.features.children())
            tmp.append(nn.ReLU().cuda())
            model.features = nn.Sequential(*tmp)
            if epoch == 0:
                model.top_layer = nn.Linear(fd, len(t_2_images_lists))
                model.top_layer.weight.data.normal_(0, 0.01)
                model.top_layer.bias.data.zero_()
                model.top_layer.cuda()
            else:
                model.top_layer = top_layer_tmp
                model.top_layer.cuda()



        #################################################################
        ########## train network with clusters as pseudo-labels #########
        #################################################################
        loss_t = train(train_dataloader, model, criterion, optimizer)
        top_layer_tmp = model.top_layer
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batch,
            num_workers=args.workers,
            pin_memory=True)
        loss_t, acc_t, output_train, target_train = feedforward(train_dataloader, model, criterion)
        loss_v, acc_v, output_test, target_test = feedforward(test_dataloader, model, criterion)
        print(target_test.shape)
        outputs.append(output_test.cpu().detach().numpy())

        targets.append(target_test.cpu().detach().numpy())



        losses_t.append(loss_t)
        losses_v.append(loss_v)
        accs_t.append(acc_t)
        accs_v.append(acc_v)
        clustering_losses_train.append(clustering_loss_train)
        clustering_losses_test.append(clustering_loss_test)



        print(
            'Epoch [{0}]| Total time: {1:.2f} s | Train loss: {2:.2f} | Val loss: {3:.2f} | C_train loss: {4: .2f} | C_val loss: {5: .2f} | Train acc: {6: .1f} | Val acc: {7: .1f} | NMI: {8: .1f} | Inclass size: '
            .format(epoch, time.time() - end,  loss_t, loss_v,
                    clustering_loss_train, clustering_loss_test, acc_t, acc_v, nmi[-1]), [len(x) for x in label_now])

        epoch_last = epoch


    '''*************************** PLOTTING **************************'''
    # plot the last t-SNE
    x_embedded = TSNE(n_components=2).fit_transform(features)
    figure, axesSubplot = plt.subplots()
    axesSubplot.scatter(x_embedded[:,0], x_embedded[:,1], c=pseudolabels_2)
    axesSubplot.set_xticks(())
    axesSubplot.set_yticks(())
    feature_tsne_file_name = 'fig%04d.png' %(epoch_last)
    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)
    plt.savefig(results_dir+'/'+feature_tsne_file_name)
    plt.close()

    # plot the loss(or modularity, Q) curve
    plt.figure()
    plt.plot(losses_t, 'y', label='train_loss')
    plt.plot(losses_v, 'r', label='test_loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    file_name = 'loss.png'

    plt.savefig(results_dir+'/'+file_name)
    plt.close()
    (pd.DataFrame(np.array(losses_t))).to_csv(results_dir+'/'+'loss_train.csv')
    (pd.DataFrame(np.array(losses_v))).to_csv(results_dir+'/'+'loss_test.csv')

    # plot the clustering loss curve
    plt.figure()
    plt.plot(clustering_losses_train, 'y', label='clustering_train_ll')
    plt.plot(clustering_losses_test, 'r', label='clustering_test_ll')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    file_name = 'clustering_loss.png'
    plt.savefig(results_dir+'/'+file_name)
    plt.close()
    (pd.DataFrame(np.array(clustering_losses_train))).to_csv(results_dir+'/'+'clustering_loss_train.csv')
    (pd.DataFrame(np.array(clustering_losses_test))).to_csv(results_dir+'/'+'clustering_loss_test.csv')


    # plot the accuracy curve
    plt.figure()
    plt.plot(accs_t, 'y', label='train_accs')
    plt.plot(accs_v, 'r', label='test_accs')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend(loc=1)
    file_name = 'acc.png'
    plt.savefig(results_dir+'/'+file_name)
    plt.close()
    (pd.DataFrame(np.array(accs_t))).to_csv(results_dir+'/'+'acc_train.csv')
    (pd.DataFrame(np.array(accs_v))).to_csv(results_dir+'/'+'acc_test.csv')


    # plot the number of members in each cluster
    c = []
    max=  0
    for i in range(len(m_list)):
        if i == 0:
            max = len(m_list[i])
        else:
            if len(m_list[i]) > max:
                max = len(m_list[i])
    for i in range(max):
        c.append([])
    for j in range(len(m_list)):
        for i in range(max):
            if len(m_list[j]) < max:
                c[i].append(0)
            else:
                c[i].append(m_list[j][i])
    plt.figure()
    clr = ['r','b','y','g','c','m','k','w']*1000
    for i in range(len(m_list[0])):
            plt.plot(c[i], color=clr[i])
    plt.xlabel('epoch')
    plt.ylabel('# of members')
    plt.legend(loc=1)

    file_name = 'members.png'
    plt.savefig(results_dir+'/'+file_name)
    plt.close()

    # plot the nmi(or ami)
    nmi[0] = 0
    plt.figure()
    plt.plot(nmi, label='AMI')
    plt.xlabel('epoch')
    plt.ylabel('AMI')
    plt.legend(loc=1)
    file_name = 'ami.png'
    plt.savefig(results_dir+'/'+file_name)
    plt.close()
    (pd.DataFrame(np.array(nmi))).to_csv(results_dir + '/' + 'ami.csv')

    # save the last labels
    (pd.DataFrame(np.array(pseudolabels_2))).to_csv(results_dir+'/'+'label.csv')
    # plot the number of clusters(k) only for louvain method
    if args.clustering == 'LV':
        plt.figure()
        plt.plot(cluster_number, label='The number of clusters')
        plt.xlabel('epoch')
        plt.ylabel('The number of clusters')
        plt.legend(loc=1)
        file_name = 'cluster'+profile+'.png'
        plt.savefig(results_dir+'/'+file_name)
        plt.close()

        # save convergent Q(modularity)
        convergent_modularity = []
        convergent_modularity.append(sum(clustering_losses_train[-11:-1])/10.0)
        (pd.DataFrame(np.array(convergent_modularity))).to_csv(results_dir + '/' + 'convQ.csv')

    # save the train and test index
    (pd.DataFrame(np.array(train_index))).to_csv(results_dir + '/' + 'train_idx.csv')
    (pd.DataFrame(np.array(test_index))).to_csv(results_dir + '/' + 'test_idx.csv')


    save_dict = {
        'pseudolabels' : np.array(pseudolabels)
    }
    sio.savemat(results_dir+'/'+'pseudolabels.mat', save_dict)
    print(np.concatenate(targets, axis=0).T)
    output_dict = {
        'outputs' : np.reshape(np.concatenate(outputs, axis=0), (args.epochs, len(test_index), args.num_cluster))
    }
    sio.savemat(results_dir+'/'+'outputs.mat', output_dict)

    target_dict = {
        'labels': np.reshape(np.concatenate(targets, axis=0).T, (args.epochs, len(test_index)))
    }
    sio.savemat(results_dir + '/' + 'labels.mat', target_dict)

    savedict = {
            'clus_features': np.array(clus_features)
        }
    sio.savemat(results_dir+'/'+'clus_features.mat', savedict)


    save_gm_params = {
        'gm_params' : np.array(params)
    }
    sio.savemat(results_dir+'/'+'gm_params.mat', save_gm_params)

    torch.save({'epoch': epoch,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict()},
               os.path.join(results_dir, 'checkpoint_epoch%04d.pth.tar' % epoch))




elif args.mode == 'test':
    # fix random seeds
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        print("/// Cuda is available ///")
        torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)


    # Front architecture
    print('Mode: {}'.format(args.mode))
    print('Architecture: {}'.format(args.arch))
    print('Data: {}'.format(args.data))
    print('DX: {}'.format(args.dx))
    print('Clustering method: {}'.format(args.clustering))
    print('')

    model = load_model(args.model)

    model.cuda()
    cudnn.benchmark = True

    # load data
    end = time.time()
    print("loading the dataset...")

    # data-type & subjects selection

    if args.data == 'atrophy':
        if args.dx == 'mci':
            transform = transforms.Compose([transforms.ToTensor,
                                            transforms.Normalize([0.4281], [0.6008])])
            dataset = corticaldataset.CorticalDataset(mode=args.mode, dx='mci', datatype='atrophy', transform=transform)
            print(dataset, len(dataset))
    elif args.data == 'wscore':
        if args.dx == 'total':
            transform = transforms.Compose([transforms.ToTensor,
                                            transforms.Normalize([0], [0.78])])
            dataset = corticaldataset.CorticalDataset(mode=args.mode, dx='total', datatype='wscore', transform=transform)

    print('Data loading time: {0:.2f} s'.format(time.time() - end))
    print(dataset.x_data.numpy().mean(), dataset.x_data.numpy().std(ddof=1))#calculate the mean and standard deviation of test data


    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=args.batch,
                                             num_workers=args.workers,
                                             pin_memory=True, )

    # train
    accs_t = []

    # making top directory's profile and some directories at each implementation

    script_dir = os.path.dirname(__file__)
    loss_dir = os.path.join(script_dir, '0_loss')
    acc_dir = os.path.join(script_dir, '6_acc')


    profile = '_test_%s_%s_%s_%s_k=%d_lr=%f_ep=%d_reassign=%d_earlystop=%s_niter=%d_fold_idx=%d' % (args.no, args.data, args.dx, args.clustering, args.num_cluster, args.lr, args.epochs, args.reassign, args.earlystop, args.niter, args.fold_idx)
    results_dir = os.path.join(script_dir, 'Results'+profile)
    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)

    # index = list(range(len(dataset)))
    # random.shuffle(index)
    # test_index = index[int(len(dataset) * 0.9):]
    # test_index.sort()
    # train_index = index[:int(len(dataset) * 0.9)]
    # train_index.sort()
    """ Train Test index """


    end = time.time()
    # remove head
    # model.top_layer = None


    #################################################################
    ########## train network with clusters as pseudo-labels #########
    #################################################################


    acc_t, predicts = test_(dataloader, model)

    accs_t.append(acc_t)
    print(predicts)
    (pd.DataFrame(np.array(predicts))).to_csv(results_dir + '/' + 'predicted_labels%s.csv' % (profile))



