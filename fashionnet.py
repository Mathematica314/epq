import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

training_data = datasets.FashionMNIST(
    root = "data",
    train = True,
    download = True,
    transform = ToTensor()
)

test_data = datasets.FashionMNIST(
    root = "data",
    train = False,
    download = True,
    transform = ToTensor()
)

batch_size = 64

train_dataloader = DataLoader(training_data,batch_size=batch_size)
test_dataloader = DataLoader(test_data,batch_size=batch_size)

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28,512),
            nn.ReLU(),
            nn.Linear(512,512),
            nn.ReLU(),
            nn.Linear(512,10)
        )
    def forward(self,x):
        return self.linear_relu_stack(self.flatten(x))

device = "cpu"
model = NeuralNetwork().to(device)
model.load_state_dict(torch.load("results/fashion.pth", weights_only=True))

loss_fn = nn.CrossEntropyLoss()
optimiser = torch.optim.SGD(model.parameters(), lr=0.001)

def train(dataloader, model, loss_fn, optimiser):
    model.train()
    for batch, (x,y) in enumerate(dataloader):
        x,y = x.to(device), y.to(device)

        pred = model(x)
        loss = loss_fn(pred,y)

        loss.backward()
        optimiser.step()
        optimiser.zero_grad()

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
    correct /= size
    print(f"Accuracy = {correct:>1f}, Loss = {test_loss:>8f}")

for t in range(500):
    print(f"Epoch {t}",end=" ")
    train(train_dataloader,model,loss_fn,optimiser)
    test(test_dataloader,model,loss_fn)
    torch.save(model.state_dict(), "results/fashion.pth")
