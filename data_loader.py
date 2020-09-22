import torch
from torch.utils.data import Dataset, DataLoader


class dict_dataset(Dataset):
    def __init__(self, input_dict):
        self.dict = input_dict
        self.feat = []
        self.edge_index = []
        self.edge_weight = []
        self.batch = []
        for key in list(self.dict.keys()):
            self.feat.append(self.dict[key][0])
            self.edge_index.append(self.dict[key][1])
            self.edge_weight.append(self.dict[key][2])
            self.batch.append(self.dict[key][3])

    def __len__(self):
        return len(list(self.dict.keys()))

    def __getitem__(self, index):
        return self.feat[index], self.edge_index[index], self.edge_weight[index], self.batch[index]
