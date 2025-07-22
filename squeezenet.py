import torchvision
import torch
import generalised
import pickle
import os
import random
from matplotlib import pyplot as plt

class SpectrogramDataset(torch.utils.data.Dataset):
    def __init__(self,seed,train):
        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            torchvision.transforms.RandomCrop([256,2048])
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
        with open(self.files[item][1],"rb") as f:
            return (self.transform(pickle.load(f).repeat(3,1,1)),self.files[item][0])

# model = torchvision.models.alexnet(weights=torchvision.models.AlexNet_Weights.DEFAULT)
# model.classifier[6] = torch.nn.Linear(4096,7)

train = SpectrogramDataset(0,True)
test = SpectrogramDataset(0,False)

model = torchvision.models.squeezenet1_1(weights=torchvision.models.SqueezeNet1_1_Weights.DEFAULT)
model.classifier._modules["1"] = torch.nn.Conv2d(512, 7, kernel_size=(1, 1))
model.num_classes=7

tr_data,te_data = generalised.do_net(None,None, 1000, shuffle=True, batch_size=32,scheduler_start=0,learning_rate=0.001,scheduler_factor=1,test_frequency=1,model=model,test_data=test,training_data=train)
plt.plot([x[1] for x in tr_data], label="train_loss")
plt.plot([x[1] for x in te_data], label="test_loss")
plt.legend()
plt.show()
plt.savefig("loss.png")
plt.plot([x[0] for x in tr_data], label="train_acc")
plt.plot([x[0] for x in te_data], label="test_acc")
plt.legend()
plt.show()
plt.savefig("acc.png")

with open("sqz.pkl","wb") as f:
    pickle.dump((tr_data,te_data),f)