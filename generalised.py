import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from matplotlib import pyplot as plt

import random
import pickle

class GeneralDataset(Dataset):
    def __init__(self,seed,train,data):
        self.composers = list(set([d[0] for d in data]))
        labeldata = [(torch.Tensor(i[1]), self.composers.index(i[0])) for i in data]
        generator = torch.Generator().manual_seed(seed)
        split = torch.utils.data.random_split(labeldata,[0.67,0.33],generator=generator)
        if train:
            self.data = split[0]
        else:
            self.data = split[1]
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
        #stack.append(nn.Sigmoid())
        self.stack = nn.Sequential(*stack)
    def forward(self,x):
        return self.stack(self.flatten(x))

def train(dataloader, model, loss_fn, optimiser,device):
    model.train()
    train_loss = 0
    correct = 0
    for batch, (x,y) in enumerate(dataloader):
        x,y = x.to(device), y.to(device)
        pred = model(x)
        loss = loss_fn(pred,y)
        train_loss += loss.item()
        loss.backward()
        optimiser.step()
        optimiser.zero_grad()
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    return (correct/len(dataloader.dataset),train_loss / (batch+1))

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

def do_net(data,net_structure,epochs,test_frequency=100,seed=0,batch_size=64,learning_rate=0.01,scheduler_factor=0.99,scheduler_start=1000,patience=3,device="cpu",verbose=True,model=None,test_data=None,training_data=None,shuffle=False):
    if training_data is None:
        training_data = GeneralDataset(train=True, seed=seed, data=data)
    if test_data is None:
        test_data = GeneralDataset(train=False, seed=seed, data=data)

    train_dataloader = DataLoader(training_data, batch_size=batch_size,shuffle=shuffle)
    test_dataloader = DataLoader(test_data, batch_size=batch_size)
    if model is None:
        model = GeneralNeuralNetwork(net_structure).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimiser = torch.optim.SGD(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimiser,scheduler_factor)

    te_results = []
    tr_results = []

    for epoch in range(epochs):
        if epoch%test_frequency == 0:
            accuracy,loss = test(test_dataloader,model,loss_fn,device)
            te_results.append((accuracy,loss))
            if verbose:
                print(f"Epoch {epoch}: Accuracy = {round(accuracy*100,2)}%, Loss = {loss}")

        accuracy,loss = train(train_dataloader,model,loss_fn,optimiser, device)
        tr_results.append((accuracy,loss))
        if epoch > scheduler_start:
            scheduler.step()
    # with open("mlp.pkl","wb") as file:
    #     pickle.dump(model,file)
    return tr_results,te_results,model

def quality(output):
    accuracies = [x[0] for x in output[80:]]
    losses = [x[1] for x in output[80:]]

    return sum([sum([x**2 for x in accuracies])/len(accuracies) - (sum(accuracies)/len(accuracies))**2,
                3*max(accuracies),
                5*sum(accuracies)/len(accuracies),
                3*accuracies[-1],
                -0.05*sum(losses)/len(losses)
                ])
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
def runtest(data,excluded_data,composers,id):
    data = [(d[0],[x for i,x in enumerate(d[1]) if i not in excluded_data]) for d in data]
    data = [d for d in data if d[0] in composers]
    tr_data,te_data,model = do_net(data, [len(data[0][1]), 128, 128, len(composers)], 100000, verbose=False,scheduler_factor=0.99995,test_frequency=1,learning_rate=0.001,patience=10,scheduler_start=0)
    with open(f"outputs/mlps/{id}_tr.pkl","wb") as f:
        pickle.dump(tr_data,f)
    with open(f"outputs/mlps/{id}_te.pkl","wb") as f:
        pickle.dump(te_data,f)
    with open(f"outputs/mlps/{id}_model.pkl","wb") as f:
        pickle.dump(model,f)

if __name__ == "__main__":
    with open("fets.pkl", "rb") as file:
        data = pickle.load(file)
    smaller_datapoints = [1, 4, 6, 13, 15, 16]
    ordered_composers = ["beethoven","bach","chopin","mozart","scarlatti","joplin","haydn"]
    # for i in range(2,8):
    #     print(i)
    #     runtest(data,[],ordered_composers[:i],f"{i}cmps_full")
    #     print(f"{i} small")
    #     runtest(data,smaller_datapoints,ordered_composers[:i],f"{i}cmps_small")
    data = [(d[0],[x for i,x in enumerate(d[1]) if i not in smaller_datapoints]) for d in data]
    tr_data,te_data,model = do_net(data, [len(data[0][1]), 128, 128, len(ordered_composers)], 50000, verbose=False,scheduler_factor=0.99994,test_frequency=1,learning_rate=0.002,patience=10,scheduler_start=0)
    plt.plot([x[1] for x in tr_data], label="train_loss")
    plt.plot([x[1] for x in te_data], label="test_loss")
    plt.legend()
    plt.show()
    plt.plot([x[0] for x in tr_data], label="train_acc")
    plt.plot([x[0] for x in te_data], label="test_acc")
    plt.ylim(0, 1)
    plt.legend()
    plt.show()
    print(max([x[0] for x in te_data]))
    import chime
    chime.success()
#0.743