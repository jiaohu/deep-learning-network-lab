import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import FashionMNIST

from model import LeNet, get_device

DATA_DIR = "./data"
MODEL_PATH = "./model.pth"
IMAGE_SIZE = 28
BATCH_SIZE = 1


def test_data_process() -> DataLoader:
    test_data = FashionMNIST(
        root=DATA_DIR,
        train=False,
        transform=transforms.Compose([transforms.Resize(size=IMAGE_SIZE), transforms.ToTensor()]),
        download=True,
    )

    return DataLoader(
        dataset=test_data,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )


def test_model(model: nn.Module, test_loader: DataLoader) -> float:
    device = get_device()
    model = model.to(device)

    test_acc = 0.0
    test_num = 0

    model.eval()
    with torch.no_grad():
        for test_data_x, test_data_y in test_loader:
            test_data_x = test_data_x.to(device)
            test_data_y = test_data_y.to(device)
            output = model(test_data_x)
            pre_lab = torch.argmax(output, dim=1)
            test_acc += torch.sum(pre_lab == test_data_y).item()
            test_num += test_data_x.size(0)

    return test_acc / test_num


def main() -> None:
    device = get_device()
    model = LeNet()
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))

    accuracy = test_model(model, test_data_process())
    print(f"Test Accuracy: {accuracy * 100:.2f}%")


if __name__ == "__main__":
    main()
