from torch.utils.data import Dataset
import pickle
import random
import torch

class OtherFeaturesDataset(Dataset):
    def __init__(self,seed,train):
        self.composers = ["bach", "beethoven", "chopin", "haydn", "joplin", "mozart", "scarlatti"]
        with open("../otfet.pkl", "rb") as file:
            data = pickle.load(file)
        random.seed(seed)
        random.shuffle(data)
        if train:
            self.data = data[150:]
        else:
            self.data = data[:150]
    def __len__(self):
        return len(self.data)
    def __getitem__(self, item):
        return torch.Tensor(self.data[item][1]), self.composers.index(self.data[item][0])