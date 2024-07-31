import time
from sklearn import mixture
import numpy as np
import torchvision.transforms as transforms
import torch.utils.data as data


class ReassignedDataset(data.Dataset):
    """A dataset where the new images labels are given in argument.
    Args:
        image_indexes (list): list of data indexes
        pseudolabels (list): list of labels for each data
        dataset (list): list of tuples with paths to images
        transform (callable, optional): a function/transform that takes in
                                        an PIL image and returns a
                                        transformed version
    """

    def __init__(self, image_indexes, pseudolabels, dataset, transform=None):
        self.imgs = self.make_dataset(image_indexes, pseudolabels, dataset)
        self.transform = transform
        self.len = len(self.imgs)

    def make_dataset(self, image_indexes, pseudolabels, dataset):
        label_to_idx = {label: idx for idx, label in enumerate(set(pseudolabels))}
        # print(label_to_idx)
        images = []
        for j, idx in enumerate(image_indexes):
            img = dataset[idx]
            pseudolabel = label_to_idx[pseudolabels[j]]
            images.append((img, pseudolabel))
        # print(images, len(images))
        return images

    def __getitem__(self, index):
        """
        Args:
            index (int): index of data
        Returns:
            tuple: (image, pseudolabel) where pseudolabel is the cluster of index datapoint
        """
        data, pseudolabel = self.imgs[index]

        return [data, pseudolabel]

    def __len__(self):
        return self.len

def cluster_assign(images_lists, dataset):
    """Creates a dataset from clustering, with clusters as labels.
    Args:
        images_lists (list of list): for each cluster, the list of image indexes
                                    belonging to this cluster
        dataset (list): initial dataset
    Returns:
        ReassignedDataset(torch.utils.data.Dataset): a dataset with clusters as
                                                     labels
    """
    assert images_lists is not None
    pseudolabels = []
    image_indexes = []
    for cluster, images in enumerate(images_lists):
        image_indexes.extend(images)
        pseudolabels.extend([cluster] * len(images))
    # print(pseudolabels)
    t = transforms.Compose([transforms.ToTensor,
                              transforms.Normalize([2.20], [0.84])])

    return ReassignedDataset(image_indexes, pseudolabels, dataset, t)


def arrange_clustering(images_lists):
    pseudolabels = []
    image_indexes = []
    for cluster, images in enumerate(images_lists):
        image_indexes.extend(images)
        pseudolabels.extend([cluster] * len(images))
    indexes = np.argsort(image_indexes)
    return np.asarray(pseudolabels)[indexes]


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