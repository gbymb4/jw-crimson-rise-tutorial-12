# -*- coding: utf-8 -*-  
"""Created on Wed Aug 20 15:18:14 2025  
@author: taske"""  
  
import torch  
import torch.nn as nn  
import torch.optim as optim  
import torchvision  
import torchvision.transforms as transforms  
 is ready for students to fill in.  
- **2. The example solution version**: All TODOs are completed so you and your students can compare their work.  
  
Both scripts use the safe structure for Windows (`if __name__ == "__main__":` and `num_workers=0`).    
You can copy/paste and run either script as a standalone `.py` file.  
  
---  
  
## 1. Solo Exercise Scaffold  
  
```python  
# -*- coding: utf-8 -*-  
"""Created on Wed Aug 20 15:18:14 2025  
@author: taske"""  
  
import torch  
import torch.nn as nn  
import torch.optim as optim  
import torchvision  
import torchvision.transforms as transforms  
import matplotlib.pyplot as plt  
  
# Baseline model  
class BaselineNet(nn.Module):  
    def __init__(self):  
        super(BaselineNet, self).__init__()  
        self.flatten = nn.Flatten()  
        self.fc1 = nn.Linear(28*28, 256)  
        self.relu1 = nn.ReLU()  
        self.fc2 = nn.Linear(256, 64)  
        self.relu2 = nn.ReLU()  
        self.fc3 = nn.Linear(64, 10)  
    def forward(self, x):  
        x = self.flatten(x)  
        x = self.relu1(self.fc1(x))  
        x = self.relu2(self.fc2(x))  
        x =import matplotlib.pyplot as plt  
  
# Device config  
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  
  
# Data loading (Windows-safe: num_workers=0)  
transform = transforms.Compose([  
    transforms.ToTensor(),  
    transforms.Normalize((0.5,), (0.5,))  
])  
trainset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)  
testset = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)  
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=0)  
testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=0)  
  
# Baseline model  
class BaselineNet(nn.Module):  
    def __init__(self):  
        super(BaselineNet, self).__init__()  
        self.flatten = nn.Flatten()  
        self.fc1 = nn.Linear(28*28, 256)  
        self.relu1 = nn.ReLU()  
        self.fc2 = nn.Linear(256, 64)  
        self.relu2 = nn.ReLU()  
        self.fc3 = nn.Linear(64, 10)  
    def forward(self, x):  
        x = self.flatten(x)  
        x = self.relu1(self.fc1(x))  
        x = self.relu2(self.fc2(x))  
        x = self.fc3(x)  
        return x  
  
# TODO 1: Create a BatchNormNet class with BatchNorm1d after each Linear layer  
class BatchNormNet(nn.Module):  
    def __init__(self):  
        super(BatchNormNet, self).__init__()  
        # HINT: use nn.BatchNorm1d after self.fc3(x)  
        return x  
  
# TODO 1: Create a BatchNormNet class which adds BatchNorm layers after each Linear layer  
class BatchNormNet(nn.Module):  
    def __init__(self):  
        super(BatchNormNet, self).__init__()  
        # TODO: Define layers, including BatchNorm1d after each Linear  
        pass  
    def forward(self, x):  
        # TODO: Implement forward pass with BatchNorm applied appropriately  
        pass  
  
def train(model, optimizer, criterion, trainloader, grad_clip=None):  
    model.train()  
    running_loss = 0.0  
    grad_norms = []  
    for inputs, targets in trainloader:  
        inputs, targets = inputs.to(next(model.parameters()).device), targets.to(next(model.parameters()).device)  
        optimizer.zero_grad()  
        outputs = model(inputs)  
        loss = criterion(outputs, targets)  
        loss.backward()  
        # TODO 2: If grad_clip is not None, clip gradients to given value  
        # TODO 3: Track total gradient norm for visualization  
        optimizer.step()  
        running_loss += loss.item() * inputs.size(0)  
    return running_loss / len(trainloader.dataset), grad_norms  
  
def evaluate(model, criterion, testloader):  
    model.eval()  
    total_loss = 0.0  
    correct = 0  
    with torch.no_grad():  
        for inputs, targets in testloader:  
            inputs, targets = inputs.to(next(model.parameters()).device), targets.to(next(model.parameters()).device)  
            outputs = model(inputs)  
            loss = criterion(outputs, targets)  
            total_loss += loss.item() * inputs.size(0)  
            _, predicted = outputs.max(1)  
            correct += predicted.eq(targets).sum().item()  
    return total_loss / len(testloader.dataset), correct / len(testloader.dataset)  
  
def main():  
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  
  
    # Data loading  
    transform = transforms.Compose([  
        transforms.ToTensor(),  
        transforms.Normalize((0.5,), (0.5,))  
    ])  
    trainset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)  
    testset = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)  
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=0)  
    testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=0)  
  
    baseline = BaselineNet().to(device)  
    # TODO 4: Initialize your BatchNormNet model and optimizer  
    batchnorm_model = None  
  
    criterion = nn.CrossEntropyLoss()  
    optimizer_base = optim.Adam(baseline.parameters(), lr=0.001)  
    # TODO 5: Create optimizer for BatchNormNet  
    optimizer_bn = None  
  
    num_epochs = 8  
    losses_base, losses_bn = [], []  
    accs_base, accs_bn = [], []  
    grad_norms_base, grad_norms_bn = [], []  
  
    print("Training baseline model...")  
    for epoch in range(num_epochs):  
        train_loss, grad_norms = train(baseline, optimizer_base, criterion, trainloader)  
        test_loss, test_acc = evaluate(baseline, criterion, testloader)  
        losses_base.append((train_loss, test_loss))  
        accs_base.append(test_acc)  
        grad_norms_base.extend(grad_norms)  
        print(f"Epoch {epoch}: Baseline Train Loss={train_loss:.4f}, Test Acc={test_acc:.4f}")  
  
    print("\nTraining BatchNormNet model with gradient clipping...")  
    # TODO 6: Train BatchNormNet, using gradient clipping (e.g., grad_clip=1.0)  
    for epoch in range(num_epochs):  
        pass  # Implement training here  
  
    # Plotting  
    plt.figure(figsize=(12,5))  
    plt.subplot(1,2,1)  
    plt.plot([l[0] for l in losses_base], label='Baseline Train')  
    plt.plot([l[1] for l in losses_base], label='Baseline Test')  
    # TODO 7: Plot BatchNormNet train/test loss curves for comparison  
    plt.title("Loss Curves")  
    plt.xlabel("Epoch")  
    plt.ylabel("Loss")  
    plt.legend()  
  
    plt.subplot(1,2,2)  
    plt.plot(accs_base, label='Baseline')  
    # TODO 8: Plot BatchNormNet test accuracy  
    plt.title("Test Accuracy")  
    plt.xlabel("Epoch")  
    plt.ylabel("Accuracy")  
    plt.legend()  
    plt.tight_layout()  
    plt.show()  
  
    # TODO 9: Plot gradient norms for both models (first 200 batches)  
    plt.figure(figsize=(6,4))  
    plt.plot(grad_norms_base[:200], label='Baseline Grad Norms')  
    # Plot BatchNormNet gradient norms here  
    plt.title("Gradient Norms (First 200 batches)")  
    plt.xlabel("Batch")  
    plt.ylabel("L2 Norm")  
    plt.legend()  
    plt.show()  
  
if __name__ == "__main__":  
    main()  