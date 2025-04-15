import torchvision
import torch
import generalised
import pickle
import os
import random

class SpectrogramDataset(torch.utils.data.Dataset):
    def __init__(self,seed,train):
        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        files = []
        numcomps = 0
        data = os.fsencode("data")
        specs = os.fsencode("spectrograms")
        for composer in os.listdir(os.path.join(data,specs)):
            for file in os.listdir(os.path.join(data,specs,composer)):
                files.append([numcomps,os.path.join(data,specs,composer,file)])
            numcomps += 1
        random.seed(seed)
        random.shuffle(files)
        if train:
            self.files = files[150:]
        else:
            self.files = files[:150]
    def __len__(self):
        return len(self.files)
    def __getitem__(self, item):
        with open(self.files[item][1],"rb") as file:
            f = pickle.load(file)
            f = f.repeat(3,1,1)
            return (self.transform(f),self.files[0])


model = torchvision.models.alexnet(weights=torchvision.models.AlexNet_Weights.DEFAULT)
model.classifier[6] = torch.nn.Linear(4096,7)

train = SpectrogramDataset(0,True)
test = SpectrogramDataset(0,False)

generalised.do_net(None,None,10000,model=model,test_data=test,training_data=train)