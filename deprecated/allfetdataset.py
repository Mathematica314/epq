import torch
from torch.utils.data import Dataset
import pickle
import random


class AllFeaturesDataset(Dataset):
    def __init__(self,seed,train):
        self.composers = ["bach", "beethoven", "chopin", "haydn", "joplin", "mozart", "scarlatti"]
        with open("../otfet.pkl", "rb") as file:
            otfetdata = pickle.load(file)
        with open("../notefreqs.pkl", "rb") as file:
            notefreqdata = pickle.load(file)
        nfpoints = []
        for composer,value in zip(notefreqdata.keys(),notefreqdata.values()):
            for file in value.values():
                row = []
                for note in "ABCDEFG":
                    for accidental in ["--","-","","#","##"]:
                        if note+accidental not in file.keys():
                            row.append(0)
                        else:
                            row.append(file[note+accidental])
                nfpoints.append((composer,row))
        data = [(nfpoints[i][0],nfpoints[i][1]+otfetdata[i][1]) for i in range(len(nfpoints))]
        random.seed(seed)
        random.shuffle(data)
        if train:
            self.data = data[150:]
        else:
            self.data = data[:150]
    def __len__(self):
        return len(self.data)
    def __getitem__(self, item):
        return torch.Tensor(self.data[item][1]),self.composers.index(self.data[item][0])