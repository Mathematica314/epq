import torch
from torch import nn
from torch.utils.data import DataLoader
import allfetdataset

training_data = allfetdataset.AllFeaturesDataset(train=True, seed=0)
test_data = allfetdataset.AllFeaturesDataset(train=False, seed=0)

batch_size = 128

train_dataloader = DataLoader(training_data,batch_size=batch_size)
test_dataloader = DataLoader(test_data,batch_size=batch_size)

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(49,512),
            nn.ReLU(),
            nn.Linear(512,512),
            nn.ReLU(),
            nn.Linear(512,512),
            nn.ReLU(),
            nn.Linear(512,7),
        )
    def forward(self,x):
        return self.linear_relu_stack(self.flatten(x))

device = "cpu"
model = NeuralNetwork().to(device)
#model.load_state_dict(torch.load("allfet.pth",weights_only=True))

loss_fn = nn.CrossEntropyLoss()
optimiser = torch.optim.SGD(model.parameters(), lr=0.0002)

def train(dataloader, model, loss_fn, optimiser):
    model.train()
    total_loss = 0
    for batch, (x,y) in enumerate(dataloader):
        x,y = x.to(device), y.to(device)

        pred = model(x)
        loss = loss_fn(pred,y)

        total_loss += loss.item()

        loss.backward()
        optimiser.step()
        optimiser.zero_grad()
    return total_loss / (batch+1)

def test(dataloader, model, loss_fn):
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
    print(f"Accuracy = {int(correct)}/{size} ({round(correct/size*100,2)}%), Loss = {test_loss}")

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser,factor=0.2)
for t in range(10000000):
    if t%100 == 0:
        print(f"Epoch {t}", end=" ")
        test(test_dataloader, model, loss_fn)
        torch.save(model.state_dict(), "../allfet.pth")
    loss = train(train_dataloader, model, loss_fn, optimiser)
    if t > 1000:
        scheduler.step(loss)
