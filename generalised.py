import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

import random
import pickle

class GeneralDataset(Dataset):
    def __init__(self,seed,train,data):
        self.composers = ["bach", "beethoven", "chopin", "haydn", "joplin", "mozart", "scarlatti"]
        labeldata = [(torch.Tensor(i[1]), self.composers.index(i[0])) for i in data]
        random.seed(seed)
        random.shuffle(labeldata)
        if train:
            self.data = labeldata[150:]
        else:
            self.data = labeldata[:150]
    def __len__(self):
        return len(self.data)
    def __getitem__(self, item):
        return self.data[item]

class GeneralNeuralNetwork(nn.Module):
    def __init__(self,structure):
        super().__init__()
        self.flatten = nn.Flatten()
        stack = []
        for i,layer in enumerate(structure[:-1]):
            stack.append(nn.Linear(layer,structure[i+1]))
            stack.append(nn.ReLU())
        stack.pop(-1)
        self.stack = nn.Sequential(*stack)
    def forward(self,x):
        return self.stack(self.flatten(x))

def train(dataloader, model, loss_fn, optimiser,device):
    model.train()
    train_loss = 0
    for batch, (x,y) in enumerate(dataloader):
        x,y = x.to(device), y.to(device)
        pred = model(x)
        loss = loss_fn(pred,y)
        train_loss += loss.item()
        loss.backward()
        optimiser.step()
        optimiser.zero_grad()
    return train_loss / (batch+1)

def test(dataloader, model, loss_fn,device):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0,0
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            test_loss += loss_fn(pred,y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    return correct/size,test_loss

def do_net(data,net_structure,epochs,test_frequency=100,seed=0,batch_size=64,learning_rate=0.01,scheduler_factor=0.99,scheduler_start=1000,device="cpu",verbose=True,model=None,test_data=None,training_data=None):
    if training_data is None:
        training_data = GeneralDataset(train=True, seed=seed, data=data)
    if test_data is None:
        test_data = GeneralDataset(train=False, seed=seed, data=data)

    train_dataloader = DataLoader(training_data, batch_size=batch_size)
    test_dataloader = DataLoader(test_data, batch_size=batch_size)
    if model is None:
        model = GeneralNeuralNetwork(net_structure).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimiser = torch.optim.SGD(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser, factor=scheduler_factor)

    results = []

    for epoch in range(epochs):
        if epoch%test_frequency == 0:
            accuracy,loss = test(test_dataloader,model,loss_fn,device)
            results.append((accuracy,loss))
            if verbose:
                print(f"Epoch {epoch}: Accuracy = {round(accuracy*100,2)}%, Loss = {loss}")
        loss = train(train_dataloader,model,loss_fn,optimiser, device)
        if epoch > scheduler_start:
            scheduler.step(loss)

    return results

def quality(output):
    accuracies = [x[0] for x in output[80:]]
    losses = [x[1] for x in output[80:]]

    return sum([sum([x**2 for x in accuracies])/len(accuracies) - (sum(accuracies)/len(accuracies))**2,
                3*max(accuracies),
                5*sum(accuracies)/len(accuracies),
                3*accuracies[-1],
                -0.05*sum(losses)/len(losses)
                ])

# with open("fets.pkl", "rb") as file:
#     data = pickle.load(file)
# sfact = 0.95
# sf_step = 0.025
#
# lrate = 0.02
# lr_step = 0.01
#
# for _ in range(10):
#     scores = []
#     for sf in (sfact+sf_step,sfact-sf_step):
#         for lr in (lrate+lr_step,lrate-lr_step):
#             print(sf,lr)
#             score = quality(do_net(data, [len(data[0][1]), 256, 256, 256, 7], 10000, scheduler_factor=sf,learning_rate=lr,verbose=False))
#             scores.append((sf,lr,score))
#             print(score)
#
#
#     sf_step /= 2
#     lr_step /= 2
#     sfact,lrate = max(scores,key=lambda x:x[2])[:2]
#     print(f"\n\n{sfact},{lrate}\n\n")
# do_net(data, [len(data[0][1]), 256, 256, 256, 7], 1000000, scheduler_factor=0.91,learning_rate=0.00059)
