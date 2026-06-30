from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import FashionMNIST

DATA_DIR = "./data"
IMAGE_SIZE = 224
BATCH_SIZE = 64
NUM_WORKERS = 0
SAMPLE_FIGURE_SIZE: tuple[float, float, Literal["in"]] = (12.0, 5.0, "in")


def load_train_data() -> FashionMNIST:
    return FashionMNIST(
        root=DATA_DIR,
        train=True,
        transform=transforms.Compose([transforms.Resize(size=IMAGE_SIZE), transforms.ToTensor()]),
        download=True,
    )


def build_train_loader(train_data: FashionMNIST) -> DataLoader:
    return DataLoader(
        dataset=train_data,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
    )


def show_sample_batch() -> None:
    train_data = load_train_data()
    train_loader = build_train_loader(train_data)
    batch_x, batch_y = next(iter(train_loader))

    images = batch_x.squeeze().numpy()
    labels = batch_y.numpy()
    class_names = train_data.classes

    print(np.unique(labels))
    print(images.shape)

    plt.figure(figsize=SAMPLE_FIGURE_SIZE)

    for index in np.arange(len(labels)):
        plt.subplot(4, 16, index + 1)
        plt.imshow(images[index, :, :], cmap=plt.cm.gray)
        plt.title(class_names[int(labels[index])], size=10)
        plt.axis("off")

    plt.subplots_adjust(wspace=0.05)
    plt.show()


if __name__ == "__main__":
    show_sample_batch()
