import torch

from torchvision import datasets, transforms

from model import BasicResidualBlock
from model import BottleneckBlock

BATCH_SIZE = 32

def get_data_loader():
    test_loader = torch.utils.data.DataLoader(
        datasets.FashionMNIST("data", train=False,
                        download=True,
                        transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                        ])),
        batch_size=BATCH_SIZE, shuffle=False,
        num_workers=0, pin_memory=True)
    return test_loader

def test_blocks(test_loader):
    for batch_idx, (data, target) in enumerate(test_loader):
        print("Input data shape: {}".format(data.shape))
        basic_block = BasicResidualBlock(1, 64)
        print("BasicResidualBlock shape: {}".format(basic_block(data).shape))
        print(basic_block)
        bottleneck_block = BottleneckBlock(1, 64)
        print("BottleNeckBlock shape: {}".format(bottleneck_block(data).shape))
        print(bottleneck_block)
        break

if __name__ == '__main__':
    test_loader = get_data_loader()
    test_blocks(test_loader)
