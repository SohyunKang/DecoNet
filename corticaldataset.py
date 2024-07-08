import torch
import numpy as np
from torch.utils.data import Dataset
from scipy import io


def wscore2atrophy(wscore):
    wscore[wscore >= 0] = 0
    return -wscore

class CorticalDataset(Dataset):
    def __init__(self, mode='train', datatype='ADNI'):
        self.mode = mode
        self.seed = 31
        np.random.seed = self.seed
        if mode == 'train':
            Wscore_ADNI_AD = np.array(io.loadmat('./data/ADNI_Wscore.mat')['Wscore'], dtype=np.float32)
            atrophy_ADNI_AD = wscore2atrophy(Wscore_ADNI_AD)
            if datatype == 'ADNI':
                Wscore = Wscore_ADNI_AD
            elif datatype == 'ADNI_mci':
                Wscore = np.array(io.loadmat('./data/ADNI_mci_Wscore.mat')['Wscore'], dtype=np.float32)
            elif datatype == 'ADNI_long':
                Wscore = np.array(io.loadmat('./data/ADNI_long_Wscore.mat')['Wscore'], dtype=np.float32)
            elif datatype == 'OASIS_atrophy':
                Wscore = np.array(io.loadmat('./data/OASIS_Wscore.mat')['Wscore'], dtype=np.float32)

            atrophy = wscore2atrophy(Wscore)
            idx = np.array(io.loadmat('./data/nan_idx_ic3.mat')['idx'], dtype=np.uint32)
            atrophy = np.delete(atrophy, idx - 1, 1)
            print(atrophy)
            if datatype == 'OASIS_atrophy':
                atrophy = (atrophy - atrophy.mean()) / atrophy.std(ddof=1)
            else:
                mean_ADNI_AD = atrophy_ADNI_AD.mean()
                std_ADNI_AD = atrophy_ADNI_AD.std(ddof=1)
                atrophy = (atrophy - mean_ADNI_AD) / std_ADNI_AD

        self.x_data = torch.from_numpy(atrophy)
        print(self.x_data, self.x_data.shape)
        self.len = self.x_data.shape[0]

    def __getitem__(self, index):
        return self.x_data[index]

    def __len__(self):
        return self.len


