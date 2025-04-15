from torch.utils.data import Dataset
import pickle
import random
import torch

class NoteFreqDataset(Dataset):
    def __init__(self,train,seed):
        self.composers = ["bach","beethoven","chopin","haydn","joplin","mozart","scarlatti"]
        with open("../notefreqs.pkl", "rb") as file:
            data = pickle.load(file)
        points = []
        for composer,value in zip(data.keys(),data.values()):
            for file in value.values():
                row = []
                for note in "ABCDEFG":
                    for accidental in ["--","-","","#","##"]:
                        if note+accidental not in file.keys():
                            row.append(0)
                        else:
                            row.append(file[note+accidental])
                points.append((composer,row))
        random.seed(seed)
        random.shuffle(points)
        if train:
            self.data = points[150:]
        else:
            self.data = points[:150]
    def __len__(self):
        return len(self.data)
    def __getitem__(self, item):
        return torch.Tensor(self.data[item][1]), self.composers.index(self.data[item][0])