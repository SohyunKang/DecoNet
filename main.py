import argparse
import time

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


import network
import clustering

from util import AverageMeter, UnifLabelSampler, load_model
import matplotlib.pyplot as plt
import os

from itertools import permutations
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
    """Making latent features for clustering, ( only feed-forward in encoder )

    :param dataloader:  whole data (both train and validation)
    :param model:       MLP with final classification layer removed
    :param N:           data length
    :return:
    """
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
    print(dataset.x_data.numpy().mean(), dataset.x_data.numpy().std(ddof=1)) #calculate the mean and standard deviation of whole data

    """ Train val index """

    assert 0 <= args.fold_idx and args.fold_idx < 10
    skf = KFold(n_splits=10, shuffle=True, random_state=args.fold_seed)
    idx_list = []
    for idx in skf.split(np.zeros(len(dataset))):
        idx_list.append(idx)
    train_index, test_index = idx_list[args.fold_idx]
    # print(train_index.shape, test_index.shape)
    # print(train_index, test_index, type(train_index), type(test_index))
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=args.batch,
                                             num_workers=args.workers,
                                             pin_memory=True, )

    # clustering method selection
    deepcluster = clustering.__dict__[args.clustering](args.num_cluster)

    # train
    losses_t = []
    losses_v = []
    accs_t = []
    accs_v = []
    clustering_losses_train = []
    clustering_losses_test = []
    m_list = []
    nmi = []
    cluster_number = []
    pseudolabels = []
    outputs = []
    targets = []
    clus_features = []

    # making top directory's profile and some directories at each implementation
    script_dir = os.path.dirname(__file__)
    profile = '_%s_%s_%s_%s_ft_%s_%s_k=%d_lr=%f_ep=%d_reassign=%d_fold_idx=%d'\
                  % (args.no, args.data, args.dx, args.level, bool(args.model), args.clustering, args.num_cluster, args.lr, args.epochs, args.reassign,  args.fold_idx)
    results_dir = os.path.join(script_dir, 'Results'+profile)


    for epoch in range(args.epochs):
        end = time.time()
        # remove head
        model.top_layer = None
        model.features = nn.Sequential(*list(model.features.children())[:-1])

        # feed-forward mlp before clustering
        features = compute_features(dataloader, model, len(dataset))    # NUMPY format

        #print('###### Deep features at Epoch [%d] ###### \n' % epoch, features, features.shape, '\n')
        clus_features.append(features)
        # for mni

        if epoch != 0:
            t_1_label = t_2_label
            t_1_images_lists = t_2_images_lists

        # cluster the features
        pseudolabels_unsorted, clustering_loss_train, clustering_loss_test, params = deepcluster.cluster(features, train_index, test_index)

        if epoch == 0:
            t_1_label = pseudolabels_unsorted
            t_1_images_lists = deepcluster.images_lists


        t_2_label = pseudolabels_unsorted

        # label stabilization for calculating accuracy
        a = list(range(len(deepcluster.images_lists)))
        permute = list(permutations(a, len(deepcluster.images_lists)))
        # print(permute)
        num_same_subject = [0]*len(permute)
        for i in range(len(permute)):
            t_2_images_lists = [[] for x in range(len(deepcluster.images_lists))]
            for j in range(len(permute[i])):
                t_2_images_lists[j] = deepcluster.images_lists[permute[i][j]]
            # print(t_2_images_lists)
            for k in range(len(permute[i])):
                num_same_subject[i] += len(set(t_1_images_lists[k]) & set(t_2_images_lists[k]))
                # print(num_same_subject)
        # print(num_same_subject)
        idx = int(np.argmax(np.array(num_same_subject)))
        # print(idx)
        t_2_images_lists = [[] for x in range(len(deepcluster.images_lists))]
        for j in range(len(permute[idx])):
            t_2_images_lists[j] = deepcluster.images_lists[permute[idx][j]]
        # print(t_1_images_lists, t_2_images_lists)

        pseudolabels_2 = [0]*len(pseudolabels_unsorted)
        for j in range(len(t_2_images_lists)):
            for i in range(len(t_2_images_lists[j])):
                pseudolabels_2[t_2_images_lists[j][i]] = j
        t_2_label = pseudolabels_2

        pseudolabels.append(pseudolabels_2)


        nmi.append(normalized_mutual_info_score(t_1_label, t_2_label))


        # for # of cluster fitting
        pseudolabels_unique = list(set(pseudolabels_2))
        # if epoch == 0:
        #     the_number_of_first_cluster = len(pseudolabels_unique)
        # else:
        #     if len(pseudolabels_unique) > the_number_of_first_cluster:
        #         the_number_of_first_cluster = len(pseudolabels_2)
        # member_ = [0] * the_number_of_first_cluster
        member_ = [0] * args.num_cluster
        cluster_number.append(len(pseudolabels_unique))
        # for # of members fitting
        # print(len(pseudolabels_unique))
        for i in range(len(pseudolabels_unique)):
            a = pseudolabels_2.count(i)
            # print(pseudolabels_2)
            # print(a)
            member_[i] = a
        m_list.append(member_)



        # assign pseudo-labels
        # print(t_2_images_lists)
        t_2_images_lists_train = copy.deepcopy(t_2_images_lists)
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

        if args.model:
            tmp = list(model.features.children())
            tmp.append(nn.ReLU().cuda())
            model.features = nn.Sequential(*tmp)
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

        # print log
        if args.clustering == 'LV':
            print('###### Epoch [{0}] ###### \n'
              'Total time: {1:.3f} s\n'
              'Clustering modularity: {2:.5f} \n'
              'MLP loss: {3:.3f}'
              .format(epoch, time.time() - end, clustering_loss_train, loss_t))
            print('')
        else:

            print(
                'Epoch [{0}]| Total time: {1:.2f} s | Train loss: {2:.2f} | Val loss: {3:.2f} | C_train loss: {4: .2f} | C_val loss: {5: .2f} | Train acc: {6: .1f} | Val acc: {7: .1f} | NMI: {8: .1f} | Inclass size: '
                .format(epoch, time.time() - end,  loss_t, loss_v,
                        clustering_loss_train, clustering_loss_test, acc_t, acc_v, nmi[-1]), member_)

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
    if not os.path.isdir(loss_dir):
        os.makedirs(loss_dir)
    plt.savefig(loss_dir+'/'+file_name)
    plt.savefig(results_dir+'/'+file_name)
    plt.close()
    (pd.DataFrame(np.array(losses_t))).to_csv(loss_dir+'/'+'loss_train%s.csv' %(profile))
    (pd.DataFrame(np.array(losses_t))).to_csv(results_dir+'/'+'loss_train.csv')
    (pd.DataFrame(np.array(losses_v))).to_csv(loss_dir+'/'+'loss_test%s.csv' %(profile))
    (pd.DataFrame(np.array(losses_v))).to_csv(results_dir+'/'+'loss_test.csv')

    # plot the clustering loss curve
    plt.figure()
    plt.plot(clustering_losses_train, 'y', label='clustering_train_ll')
    plt.plot(clustering_losses_test, 'r', label='clustering_test_ll')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    file_name = 'clustering_loss.png'
    if not os.path.isdir(c_loss_dir):
        os.makedirs(c_loss_dir)
    plt.savefig(c_loss_dir+'/'+file_name)
    plt.savefig(results_dir+'/'+file_name)
    plt.close()
    (pd.DataFrame(np.array(clustering_losses_train))).to_csv(c_loss_dir+'/'+'clustering_loss_train%s.csv' %(profile))
    (pd.DataFrame(np.array(clustering_losses_train))).to_csv(results_dir+'/'+'clustering_loss_train.csv')
    (pd.DataFrame(np.array(clustering_losses_test))).to_csv(c_loss_dir+'/'+'clustering_loss_test%s.csv' %(profile))
    (pd.DataFrame(np.array(clustering_losses_test))).to_csv(results_dir+'/'+'clustering_loss_test.csv')


    # plot the accuracy curve
    plt.figure()
    plt.plot(accs_t, 'y', label='train_accs')
    plt.plot(accs_v, 'r', label='test_accs')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend(loc=1)
    file_name = 'acc.png'
    if not os.path.isdir(acc_dir):
        os.makedirs(acc_dir)
    plt.savefig(acc_dir+'/'+file_name)
    plt.savefig(results_dir+'/'+file_name)
    plt.close()
    (pd.DataFrame(np.array(accs_t))).to_csv(acc_dir+'/'+'acc_train%s.csv' %(profile))
    (pd.DataFrame(np.array(accs_t))).to_csv(results_dir+'/'+'acc_train.csv')
    (pd.DataFrame(np.array(accs_v))).to_csv(acc_dir+'/'+'acc_test%s.csv' %(profile))
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
    if not os.path.isdir(members_dir):
        os.makedirs(members_dir)
    plt.savefig(members_dir+'/'+file_name)
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
    if not os.path.isdir(nmi_dir):
        os.makedirs(nmi_dir)
    plt.savefig(nmi_dir+'/'+file_name)
    plt.savefig(results_dir+'/'+file_name)
    plt.close()
    (pd.DataFrame(np.array(nmi))).to_csv(nmi_dir + '/' + 'ami%s.csv' % (profile))
    (pd.DataFrame(np.array(nmi))).to_csv(results_dir + '/' + 'ami.csv')

    # save the last labels
    if not os.path.isdir(label_dir):
        os.makedirs(label_dir)
    (pd.DataFrame(np.array(pseudolabels_2))).to_csv(label_dir+'/'+'label%s.csv' %(profile))
    (pd.DataFrame(np.array(pseudolabels_2))).to_csv(results_dir+'/'+'label.csv')
    # plot the number of clusters(k) only for louvain method
    if args.clustering == 'LV':
        plt.figure()
        plt.plot(cluster_number, label='The number of clusters')
        plt.xlabel('epoch')
        plt.ylabel('The number of clusters')
        plt.legend(loc=1)
        file_name = 'cluster'+profile+'.png'
        if not os.path.isdir(clusters_dir):
            os.makedirs(clusters_dir)
        plt.savefig(clusters_dir+'/'+file_name)
        plt.savefig(results_dir+'/'+file_name)
        plt.close()

        # save convergent Q(modularity)
        convergent_modularity = []
        convergent_modularity.append(sum(clustering_losses_train[-11:-1])/10.0)
        if not os.path.isdir(convQ_dir):
            os.makedirs(convQ_dir)
        (pd.DataFrame(np.array(convergent_modularity))).to_csv(convQ_dir + '/' + 'convQ%s.csv' % (profile))
        (pd.DataFrame(np.array(convergent_modularity))).to_csv(results_dir + '/' + 'convQ.csv')

    # save the train and test index
    if not os.path.isdir(idx_dir):
        os.makedirs(idx_dir)
    (pd.DataFrame(np.array(train_index))).to_csv(idx_dir+'/'+'train_idx%s.csv' %(profile))
    (pd.DataFrame(np.array(train_index))).to_csv(results_dir + '/' + 'train_idx.csv')
    (pd.DataFrame(np.array(test_index))).to_csv(idx_dir+'/'+'test_idx%s.csv' %(profile))
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



