import os
import json

import numpy as np
import torch
from torch.utils.data.sampler import Sampler
from itertools import permutations
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import pandas as pd
from scipy import io
from sklearn import mixture
from sklearn.mixture._gaussian_mixture import _compute_precision_cholesky

import network


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

    return np.array(label_now), images_lists_now


def load_model(path):
    if os.path.isfile(path):
        print("=> loading checkpoint '{}'".format(path))
        checkpoint = torch.load(path)
        # size of the top layer
        N = checkpoint['state_dict']['top_layer.bias'].size()
        print(N)
        # build skeleton of the model
        model = network.mlp(input_dim=1193, output_dim=N[0])
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

def load_gmm(path):
    gmm_params = io.loadmat(path)['gm_params'][0,0][0][0]
    clf = mixture.GaussianMixture(n_components=len(gmm_params[0]), covariance_type='full')
    clf.weights_ = gmm_params[0]
    clf.means_ = gmm_params[1]
    clf.covariances_ = gmm_params[2]
    clf.precisions_cholesky_ = _compute_precision_cholesky(clf.covariances_, 'full')
    return clf

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

def plotandsave(results_dir, results, train_index, test_index, gmm_params, model, optimizer, epoch):
    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)

    with open(results_dir + '/' + 'results.json', 'w') as f:
        json.dump(results, f)
    x_embedded = TSNE(n_components=2).fit_transform(results['features'][f'epoch_{epoch + 1}'])
    figure, axesSubplot = plt.subplots()
    axesSubplot.scatter(x_embedded[:, 0], x_embedded[:, 1], c=results['labels'][f'epoch_{epoch + 1}'])
    axesSubplot.set_xticks(())
    axesSubplot.set_yticks(())
    feature_tsne_file_name = 'fig%04d.png' % (epoch)
    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)
    plt.savefig(results_dir + '/' + feature_tsne_file_name)
    plt.close()

    plt.figure()
    plt.plot(results['train_loss'], 'y', label='train_loss')
    plt.plot(results['val_loss'], 'r', label='test_loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    file_name = 'loss.png'
    plt.savefig(results_dir + '/' + file_name)
    plt.close()

    plt.figure()
    plt.plot(results['train_accuracy'], 'y', label='train_accs')
    plt.plot(results['val_accuracy'], 'r', label='test_accs')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend(loc=1)
    file_name = 'acc.png'
    plt.savefig(results_dir + '/' + file_name)
    plt.close()

    plt.figure()
    plt.plot(results['train_ami'], 'y', label='train_ami')
    plt.plot(results['val_ami'], 'r', label='test_ami')
    plt.xlabel('epoch')
    plt.ylabel('AMI')
    plt.legend(loc=1)
    file_name = 'ami.png'
    plt.savefig(results_dir + '/' + file_name)
    plt.close()

    # save the train and test index
    (pd.DataFrame(np.array(train_index))).to_csv(results_dir + '/' + 'train_idx.csv')
    (pd.DataFrame(np.array(test_index))).to_csv(results_dir + '/' + 'test_idx.csv')

    save_gm_params = {
        'gm_params': np.array(gmm_params)
    }
    io.savemat(results_dir + '/' + 'gm_params.mat', save_gm_params)

    torch.save({'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict()},
               os.path.join(results_dir, 'checkpoint_epoch%04d.pth.tar' % epoch))

