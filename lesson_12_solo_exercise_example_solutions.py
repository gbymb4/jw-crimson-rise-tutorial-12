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
        x = self.fc3(x)  
        return x  
  
# BatchNormNet with BatchNorm layers after each Linear  
class BatchNormNet(nn.Module):  
    def __init__(self):  
        super(BatchNormNet, self).__init__()  
        self.flatten = nn.Flatten()  
        self.fc1 = nn.Linear(28*28, 256)  
        self.bn1 = nn.BatchNorm1d(256)  
        self.relu1 = nn.ReLU()  
        self.fc2 = nn.Linear(256, 64)  
        self.bn2 = nn.BatchNorm1d(64)  
        self.relu2 = nn.ReLU()  
        self.fc3 = nn.Linear(64, 10)  
    def forward(self, x):  
        x = self.flatten(x)  
        x = self.fc1(x)  
        x = self.bn1(x)  
        x = self.relu1(x)  
        x = self.fc2(x)  
        x = self.bn2(x)  
        x = self.relu2(x)  
        x = self.fc3(x)  
        return x  
  
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
        # Gradient clipping if specified  
        if grad_clip is not None:  
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)  
        # Track total gradient norm  
        total_norm = 0.0  
        for p in model.parameters():  
            if p.grad is not None:  
                param_norm = p.grad.data.norm(2)  
                total_norm += param_norm.item() ** 2  
        grad_norms.append(total_norm ** 0.5)  
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
    batchnorm_model = BatchNormNet().to(device)  
  
    criterion = nn.CrossEntropyLoss()  
    optimizer_base = optim.Adam(baseline.parameters(), lr=0.001)  
    optimizer_bn = optim.Adam(batchnorm_model.parameters(), lr=0.001)  
  
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
    for epoch in range(num_epochs):  
        train_loss, grad_norms = train(batchnorm_model, optimizer_bn, criterion, trainloader, grad_clip=1.0)  
        test_loss, test_acc = evaluate(batchnorm_model, criterion, testloader)  
        losses_bn.append((train_loss, test_loss))  
        accs_bn.append(test_acc)  
        grad_norms_bn.extend(grad_norms)  
        print(f"Epoch {epoch}: BatchNormNet Train Loss={train_loss:.4f}, Test Acc={test_acc:.4f}")  
  
    # Plotting  
    plt.figure(figsize=(12,5))  
    plt.subplot(1,2,1)  
    plt.plot([l[0] for l in losses_base], label='Baseline Train')  
    plt.plot([l[1] for l in losses_base], label='Baseline Test')  
    plt.plot([l[0] for l in losses_bn], label='BatchNormNet Train')  
    plt.plot([l[1] for l in losses_bn], label='BatchNormNet Test')  
    plt.title("Loss Curves")  
    plt.xlabel("Epoch")  
    plt.ylabel("Loss")  
    plt.legend()  
  
    plt.subplot(1,2,2)  
    plt.plot(accs_base, label='Baseline')  
    plt.plot(accs_bn, label='BatchNormNet')  
    plt.title("Test Accuracy")  
    plt.xlabel("Epoch")  
    plt.ylabel("Accuracy")  
    plt.legend()  
    plt.tight_layout()  
    plt.show()  
  
    plt.figure(figsize=(6,4))  
    plt.plot(grad_norms_base[:200], label='Baseline Grad Norms')  
    plt.plot(grad_norms_bn[:200], label='BatchNormNet Grad Norms')  
    plt.title("Gradient Norms (First 200 batches)")  
    plt.xlabel("Batch")  
    plt.ylabel("L2 Norm")  
    plt.legend()  
    plt.show()  
  
if __name__ == "__main__":  
    main()  