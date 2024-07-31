import os
import pickle

import numpy as np
import torch
from torch.utils.data.sampler import Sampler
from itertools import permutations

import models


def label_stab(images_lists_unmatched, images_lists_prev, label_unmatched):
    """Stabilize labels across epochs for classification."""
    num_cluster = len(images_lists_unmatched)
    a = list(range(num_cluster))
    permute = list(permutations(a))
    num_same_subject = [0] * len(permute)
    for i in range(len(permute)):
        images_lists_perm = [[] for x in range(num_cluster)]
        for j in range(num_cluster):
            images_lists_perm[j] = images_lists_unmatched[permute[i][j]]
        for k in range(num_cluster):
            num_same_subject[i] += len(set(images_lists_prev[k]) & set(images_lists_perm[k]))
    perm_idx = int(np.argmax(np.array(num_same_subject)))   # select a proper permutation

    images_lists_now = [[] for x in range(num_cluster)]
    for j in range(num_cluster):
        images_lists_now[j] = images_lists_unmatched[permute[perm_idx][j]]

    label_now = [0] * len(label_unmatched)
    for j in range(num_cluster):
        for k in range(len(images_lists_now[j])):
            label_now[images_lists_now[j][k]] = j

    return label_now, images_lists_now


def load_model(path):
    """Loads model and return it without DataParallel table."""
    if os.path.isfile(path):
        print("=> loading checkpoint '{}'".format(path))
        checkpoint = torch.load(path)

        # size of the top layer
        N = checkpoint['state_dict']['top_layer.bias'].size()

        # build skeleton of the model
        model = models.__dict__[checkpoint['arch']](dim=1193, level='ic3',out=int(N[0]))

        # deal with a dataparallel table
        def rename_key(key):
            if not 'module' in key:
                return key
            return ''.join(key.split('.module'))

        checkpoint['state_dict'] = {rename_key(key): val
                                    for key, val
                                    in checkpoint['state_dict'].items()}

        # load weights
        model.load_state_dict(checkpoint['state_dict'])
        print("Loaded")
    else:
        model = None
        print("=> no checkpoint found at '{}'".format(path))
    return model


class UnifLabelSampler(Sampler):
    """Samples elements uniformely accross pseudolabels.
    """
    def __init__(self, N, images_lists):
        self.N = N
        self.images_lists = images_lists
        self.indexes = self.generate_indexes_epoch()

    def generate_indexes_epoch(self):
        nmb_non_empty_clusters = 0
        for i in range(len(self.images_lists)):
            if len(self.images_lists[i]) != 0:
                nmb_non_empty_clusters += 1

        size_per_pseudolabel = int(self.N / nmb_non_empty_clusters) + 1
        res = np.array([])
        for i in range(len(self.images_lists)):
            # skip empty clusters
            if len(self.images_lists[i]) == 0:
                continue
            indexes = np.random.choice(
                self.images_lists[i],
                size_per_pseudolabel,
                replace=(len(self.images_lists[i]) <= size_per_pseudolabel)
            )
            res = np.concatenate((res, indexes))
        np.random.shuffle(res)
        res = list(res.astype('int'))
        if len(res) >= self.N:
            return res[:self.N]
        res += res[: (self.N - len(res))]
        return res
    def __iter__(self):
        return iter(self.indexes)
    def __len__(self):
        return len(self.indexes)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def plotandsave():
    # plot the last t-SNE
    x_embedded = TSNE(n_components=2).fit_transform(features)
    figure, axesSubplot = plt.subplots()
    axesSubplot.scatter(x_embedded[:, 0], x_embedded[:, 1], c=pseudolabels_2)
    axesSubplot.set_xticks(())
    axesSubplot.set_yticks(())
    feature_tsne_file_name = 'fig%04d.png' % (epoch)
    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)
    plt.savefig(results_dir + '/' + feature_tsne_file_name)
    plt.close()

    # plot the loss(or modularity, Q) curve
    plt.figure()
    plt.plot(losses_t, 'y', label='train_loss')
    plt.plot(losses_v, 'r', label='test_loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    file_name = 'loss.png'

    plt.savefig(results_dir + '/' + file_name)
    plt.close()
    (pd.DataFrame(np.array(losses_t))).to_csv(results_dir + '/' + 'loss_train.csv')
    (pd.DataFrame(np.array(losses_v))).to_csv(results_dir + '/' + 'loss_test.csv')

    # plot the clustering loss curve
    plt.figure()
    plt.plot(clustering_losses_train, 'y', label='clustering_train_ll')
    plt.plot(clustering_losses_test, 'r', label='clustering_test_ll')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    file_name = 'clustering_loss.png'
    plt.savefig(results_dir + '/' + file_name)
    plt.close()
    (pd.DataFrame(np.array(clustering_losses_train))).to_csv(results_dir + '/' + 'clustering_loss_train.csv')
    (pd.DataFrame(np.array(clustering_losses_test))).to_csv(results_dir + '/' + 'clustering_loss_test.csv')

    # plot the accuracy curve
    plt.figure()
    plt.plot(accs_t, 'y', label='train_accs')
    plt.plot(accs_v, 'r', label='test_accs')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend(loc=1)
    file_name = 'acc.png'
    plt.savefig(results_dir + '/' + file_name)
    plt.close()
    (pd.DataFrame(np.array(accs_t))).to_csv(results_dir + '/' + 'acc_train.csv')
    (pd.DataFrame(np.array(accs_v))).to_csv(results_dir + '/' + 'acc_test.csv')

    # plot the number of members in each cluster
    c = []
    max = 0
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
    clr = ['r', 'b', 'y', 'g', 'c', 'm', 'k', 'w'] * 1000
    for i in range(len(m_list[0])):
        plt.plot(c[i], color=clr[i])
    plt.xlabel('epoch')
    plt.ylabel('# of members')
    plt.legend(loc=1)

    file_name = 'members.png'
    plt.savefig(results_dir + '/' + file_name)
    plt.close()

    # plot the nmi(or ami)
    nmi[0] = 0
    plt.figure()
    plt.plot(nmi, label='AMI')
    plt.xlabel('epoch')
    plt.ylabel('AMI')
    plt.legend(loc=1)
    file_name = 'ami.png'
    plt.savefig(results_dir + '/' + file_name)
    plt.close()
    (pd.DataFrame(np.array(nmi))).to_csv(results_dir + '/' + 'ami.csv')

    # save the last labels
    (pd.DataFrame(np.array(pseudolabels_2))).to_csv(results_dir + '/' + 'label.csv')
    # plot the number of clusters(k) only for louvain method

    # save the train and test index
    (pd.DataFrame(np.array(train_index))).to_csv(results_dir + '/' + 'train_idx.csv')
    (pd.DataFrame(np.array(test_index))).to_csv(results_dir + '/' + 'test_idx.csv')

    save_dict = {
        'pseudolabels': np.array(pseudolabels)
    }
    sio.savemat(results_dir + '/' + 'pseudolabels.mat', save_dict)
    print(np.concatenate(targets, axis=0).T)
    output_dict = {
        'outputs': np.reshape(np.concatenate(outputs, axis=0), (args.epochs, len(test_index), args.num_cluster))
    }
    sio.savemat(results_dir + '/' + 'outputs.mat', output_dict)

    target_dict = {
        'labels': np.reshape(np.concatenate(targets, axis=0).T, (args.epochs, len(test_index)))
    }
    sio.savemat(results_dir + '/' + 'labels.mat', target_dict)

    savedict = {
        'clus_features': np.array(clus_features)
    }
    sio.savemat(results_dir + '/' + 'clus_features.mat', savedict)

    save_gm_params = {
        'gm_params': np.array(params)
    }
    sio.savemat(results_dir + '/' + 'gm_params.mat', save_gm_params)

    torch.save({'epoch': epoch,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict()},
               os.path.join(results_dir, 'checkpoint_epoch%04d.pth.tar' % epoch))


def learning_rate_decay(optimizer, t, lr_0):
    for param_group in optimizer.param_groups:
        lr = lr_0 / np.sqrt(1 + lr_0 * param_group['weight_decay'] * t)
        param_group['lr'] = lr


class Logger(object):
    """ Class to update every epoch to keep trace of the results
    Methods:
        - log() log and save
    """

    def __init__(self, path):
        self.path = path
        self.data = []

    def log(self, train_point):
        self.data.append(train_point)
        with open(os.path.join(self.path), 'wb') as fp:
            pickle.dump(self.data, fp, -1)

