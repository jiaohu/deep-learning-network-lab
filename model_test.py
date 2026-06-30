import torch
import torch.utils.data as Data
from torchvision import transforms
from torchvision.datasets import FashionMNIST

from model import LeNet, get_device


def test_data_process():
    test_data = FashionMNIST(
        root="./data",
        train=False,
        transform=transforms.Compose([transforms.Resize(size=28), transforms.ToTensor()]),
        download=True,
    )


    test_loader = Data.DataLoader(
        dataset=test_data,
        batch_size=1,
        shuffle=True,
        num_workers=0,
    )

    return test_loader


def test_model(model, test_loader):
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

    test_acc = test_acc / test_num
    print(f"Test Accuracy: {test_acc * 100:.2f}%")

if __name__ == "__main__":
    model = LeNet()
    model.load_state_dict(torch.load("./model.pth"))

    test_loader = test_data_process()
    test_model(model, test_loader)