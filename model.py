import torch
from torch import nn
import torch.nn.functional as F


class LeNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, padding=2)
        self.sig = nn.Sigmoid()
        self.s1 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        self.s2 = nn.AvgPool2d(kernel_size=2, stride=2)

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_features=400, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=84)
        self.fc3 = nn.Linear(in_features=84, out_features=10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.sig(self.conv1(x))
        x = self.s1(x)
        x = self.sig(self.conv2(x))
        x = self.s2(x)
        x = self.flatten(x)
        x = self.sig(self.fc1(x))
        x = self.sig(self.fc2(x))
        x = self.fc3(x)
        return x

class AlexNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.ReLU = nn.ReLU()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=96, kernel_size=11, stride=4)
        self.s1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv2 = nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, padding=2)
        self.s2 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv3 = nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, padding=1)
        self.s3 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_features=6*6*256, out_features=4096)
        self.fc2 = nn.Linear(in_features=4096, out_features=4096)
        self.fc3 = nn.Linear(in_features=4096, out_features=10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.ReLU(self.conv1(x))
        x = self.s1(x)
        x = self.ReLU(self.conv2(x))
        x = self.s2(x)
        x = self.ReLU(self.conv3(x))
        x = self.ReLU(self.conv4(x))
        x = self.ReLU(self.conv5(x))
        x = self.s3(x)

        x = self.flatten(x)
        x = self.ReLU(self.fc1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.ReLU(self.fc2(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.fc3(x)

        return x


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def main() -> None:
    from torchsummary import summary

    device = get_device()
    model = LeNet().to(device)
    print(summary(model, (1, 28, 28)))


def main_alexnet() -> None:
    from torchsummary import summary
    device = get_device()
    model = AlexNet().to(device)
    print(summary(model, (1, 227, 227)))

if __name__ == "__main__":
    main_alexnet()
