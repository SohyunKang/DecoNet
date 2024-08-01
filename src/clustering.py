import time
from sklearn import mixture
import numpy as np
import torchvision.transforms as transforms
import torch.utils.data as data
from scipy import io


class ReassignedDataset(data.Dataset):
    """A dataset where the new images labels are given in argument.
    Args:
        dataset (array): array of atrophy
        pseudolabels (list): list of labels for whole dataset
        """
    def __init__(self, dataset, pseudolabels):
        self.data = self.make_dataset(dataset, pseudolabels)
        self.len = len(self.data)
    def make_dataset(self, dataset, pseudolabels):
        data = []
        for img, pseudolabel in zip(dataset, pseudolabels):
            data.append((img, pseudolabel))
        return data
    def __getitem__(self, index):
        img, pseudolabel = self.data[index]
        return [img, pseudolabel]
    def __len__(self):
        return self.len


def run_gmm(x, train_idx, test_idx, k):
    clf = mixture.GaussianMixture(n_components=k, covariance_type='full')
    label_array = np.ones((len(x)))
    # train gmm: only using train dataset
    clf.fit(x[train_idx])
    label_train = clf.predict(x[train_idx])
    label_test = clf.predict(x[test_idx])
    gmm_params = {
        'weights': clf.weights_,
        'means': clf.means_,
        'cov': clf.covariances_
    }
    label_array[train_idx] = list(label_train)
    label_array[test_idx] = list(label_test)
    return label_array, gmm_params



class GMM(object):
    def __init__(self, k):
        self.k = k

    def cluster(self, data, train_idx, test_idx):
        """Performs Gaussian Mixture Model-based clustering.
            Args:
                data (np.array N * dim): data to cluster
        """
        label_array, params = run_gmm(data, train_idx, test_idx, self.k)
        self.images_lists = [[] for i in range(self.k)]
        for i in range(len(data)):
            self.images_lists[int(label_array[i])].append(i)

        return label_array, params