import copy
import time

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.utils.data as Data
from torch import optim, nn
from torchvision import transforms
from torchvision.datasets import FashionMNIST

from model import LeNet, get_device


def train_val_data_process():
    train_data = FashionMNIST(
        root="./data",
        train=True,
        transform=transforms.Compose([transforms.Resize(size=28), transforms.ToTensor()]),
        download=True,
    )

    train_data, val_data = Data.random_split(train_data, [round(0.8 * len(train_data)), round(0.2 * len(train_data))])
    train_loader = Data.DataLoader(
        dataset=train_data,
        batch_size=32,
        shuffle=True,
        num_workers=2,
    )

    val_loader = Data.DataLoader(
        dataset=val_data,
        batch_size=32,
        shuffle=True,
        num_workers=2,
    )

    return train_loader, val_loader

def train_model(model, train_loader, val_loader, num_epochs):
    device = get_device()

    # 优化器, 学习率0.001
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 交叉熵损失函数
    criterion = nn.CrossEntropyLoss()

    model = model.to(device)

    # 复制模型参数
    best_model_wts = copy.deepcopy(model.state_dict())

    best_acc = 0.0
    train_loss_all = []
    val_loss_all = []

    train_acc_all = []
    val_acc_all = []

    since = time.time()

    for epoch in range(num_epochs):
        print(f"Epoch {epoch}/{num_epochs-1}")
        print("-"*10)

        train_loss = 0.0
        train_acc = 0.0

        val_loss = 0.0
        val_acc = 0.0

        train_num = 0
        val_num = 0

        model.train()
        for step, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            pre_labels = torch.argmax(outputs, dim=1)
            optimizer.zero_grad()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)
            train_acc += (pre_labels == labels).sum().item()
            train_num += inputs.size(0)

        model.eval()
        for step, (inputs, labels) in enumerate(val_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)

            pre_labels = torch.argmax(outputs, dim=1)

            val_loss += criterion(outputs, labels).item() * inputs.size(0)
            val_num += inputs.size(0)
            val_acc += (pre_labels == labels).float().sum().item()

        train_loss_all.append(train_loss / train_num)
        val_loss_all.append(val_loss / val_num)

        train_acc_all.append(train_acc / train_num)
        val_acc_all.append(val_acc / val_num)

        print("{} train loss:{:.4f} train acc:{:.4f} val loss:{:.4f} val acc:{:.4f}".format(epoch, train_loss_all[-1], train_acc_all[-1], val_loss_all[-1], val_acc_all[-1]))

        if val_acc_all[-1] > best_acc:
            best_acc = val_acc_all[-1]
            best_model_wts = copy.deepcopy(model.state_dict())

        time_elapsed = time.time() - since
        print("Training complete in {:.0f}m {:.0f}s".format(time_elapsed // 60, time_elapsed % 60))

    model.load_state_dict(best_model_wts)
    torch.save(model.state_dict(), "./model.pth")

    train_process = pd.DataFrame({
        "epoch": range(num_epochs),
        "train_loss": train_loss_all,
        "train_acc": train_acc_all,
        "val_loss": val_loss_all,
        "val_acc": val_acc_all,
    })

    return train_process


def matplotlib_acc_show(train_process: pd.DataFrame):
    plt.figure(figsize=(12, 4))
    plt.subplot(1,2,1)
    plt.plot(train_process["epoch"], train_process["train_acc"], label="train_acc")
    plt.plot(train_process["epoch"], train_process["val_acc"], label="val_acc")
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("accuracy")

    plt.subplot(1,2,2)
    plt.plot(train_process["epoch"], train_process["train_loss"], label="train_loss")
    plt.plot(train_process["epoch"], train_process["val_loss"], label="val_loss")
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.show()

if __name__ == "__main__":
    train_loader, val_loader = train_val_data_process()
    model = LeNet()
    train_process = train_model(model, train_loader, val_loader, num_epochs=20)
    matplotlib_acc_show(train_process)