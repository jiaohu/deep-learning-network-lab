import copy
import time
from typing import Literal

import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import FashionMNIST

from model import VGG16, get_device

DATA_DIR = "./data"
MODEL_PATH = "./vgg16_model.pth"
IMAGE_SIZE = 224
BATCH_SIZE = 32
NUM_WORKERS = 2
LEARNING_RATE = 0.001
TRAIN_RATIO = 0.8
NUM_EPOCHS = 20
METRICS_FIGURE_SIZE: tuple[float, float, Literal["in"]] = (12.0, 4.0, "in")


def train_val_data_process() -> tuple[DataLoader, DataLoader]:
    train_data = FashionMNIST(
        root=DATA_DIR,
        train=True,
        transform=transforms.Compose([transforms.Resize(size=IMAGE_SIZE), transforms.ToTensor()]),
        download=True,
    )

    train_size = round(TRAIN_RATIO * len(train_data))
    val_size = len(train_data) - train_size
    train_data, val_data = random_split(train_data, [train_size, val_size])

    train_loader = DataLoader(
        dataset=train_data,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
    )

    val_loader = DataLoader(
        dataset=val_data,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    return train_loader, val_loader


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int,
) -> pd.DataFrame:
    device = get_device()

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    model = model.to(device)

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    train_loss_all: list[float] = []
    val_loss_all: list[float] = []
    train_acc_all: list[float] = []
    val_acc_all: list[float] = []

    since = time.time()

    for epoch in range(num_epochs):
        print(f"Epoch {epoch}/{num_epochs - 1}")
        print("-" * 10)

        train_loss = 0.0
        train_acc = 0.0
        val_loss = 0.0
        val_acc = 0.0
        train_num = 0
        val_num = 0

        model.train()
        for inputs, labels in train_loader:
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
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                pre_labels = torch.argmax(outputs, dim=1)

                val_loss += criterion(outputs, labels).item() * inputs.size(0)
                val_num += inputs.size(0)
                val_acc += (pre_labels == labels).sum().item()

        train_loss_all.append(train_loss / train_num)
        val_loss_all.append(val_loss / val_num)
        train_acc_all.append(train_acc / train_num)
        val_acc_all.append(val_acc / val_num)

        print(
            f"{epoch} train loss:{train_loss_all[-1]:.4f} "
            f"train acc:{train_acc_all[-1]:.4f} "
            f"val loss:{val_loss_all[-1]:.4f} "
            f"val acc:{val_acc_all[-1]:.4f}"
        )

        if val_acc_all[-1] > best_acc:
            best_acc = val_acc_all[-1]
            best_model_wts = copy.deepcopy(model.state_dict())

        time_elapsed = time.time() - since
        print(f"Elapsed time: {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")

    # model.load_state_dict(best_model_wts)
    torch.save(best_model_wts, MODEL_PATH)

    print(f"Training complete. Best val acc: {best_acc:.4f}")

    return pd.DataFrame(
        {
            "epoch": range(num_epochs),
            "train_loss": train_loss_all,
            "train_acc": train_acc_all,
            "val_loss": val_loss_all,
            "val_acc": val_acc_all,
        }
    )


def matplotlib_acc_show(train_process: pd.DataFrame) -> None:
    plt.figure(figsize=METRICS_FIGURE_SIZE)
    plt.subplot(1, 2, 1)
    plt.plot(train_process["epoch"], train_process["train_acc"], label="train_acc")
    plt.plot(train_process["epoch"], train_process["val_acc"], label="val_acc")
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("accuracy")

    plt.subplot(1, 2, 2)
    plt.plot(train_process["epoch"], train_process["train_loss"], label="train_loss")
    plt.plot(train_process["epoch"], train_process["val_loss"], label="val_loss")
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.show()


def main() -> None:
    train_loader, val_loader = train_val_data_process()
    model = VGG16()
    train_process = train_model(model, train_loader, val_loader, num_epochs=NUM_EPOCHS)
    matplotlib_acc_show(train_process)


if __name__ == "__main__":
    main()
