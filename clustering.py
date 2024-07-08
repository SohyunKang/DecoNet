import time
from sklearn import mixture
import numpy as np
import torchvision.transforms as transforms



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
    cluster_array = np.ones((len(x)))
    clf.fit(x[train_idx])
    cluster_train = clf.predict(x[train_idx])
    cluster_test = clf.predict(x[test_idx])
    ll_train = clf.score(x[train_idx])
    ll_test = clf.score(x[test_idx])
    params = {}
    params['weights'] = clf.weights_
    params['means'] = clf.means_
    params['cov'] = clf.covariances_
    cluster_array[train_idx] = list(cluster_train)
    cluster_array[test_idx] = list(cluster_test)
    return cluster_array, ll_train, ll_test, params


class GMM(object):
    def __init__(self, k):
        self.k = k

    def cluster(self, data, train_idx, test_idx):
        """Performs k-means clustering.
            Args:
                x_data (np.array N * dim): data to cluster
        """
        I, loss_train, loss_test, params = run_gmm(data, train_idx, test_idx, self.k)
        self.images_lists = [[] for i in range(self.k)]
        for i in range(len(data)):
            self.images_lists[int(I[i])].append(i)

        return I, loss_train, loss_test, params