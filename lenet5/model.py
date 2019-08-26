import torch.nn as nn
import torch.nn.functional as F

class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        # Input data shape (1, 28, 28)
        # in_channel=1, out_channel=6, kernel_size=5, stride=1, padding=2
        self.conv1 = nn.Conv2d(1, 6, 5, 1, 2)
        # in_channel=6, out_channel=16, kernel_size=5, stride=1
        self.conv2 = nn.Conv2d(6, 16, 5, 1)
        # in_channel=6, out_channel=16, kernel_size=5, stride=1
        self.fc1 = nn.Linear(16*5*5, 84)
        self.fc2 = nn.Linear(84, 10)

    def forward(self, x):
        z1 = F.relu(self.conv1(x))
        x = F.max_pool2d(z1, 2, 2)
        z2 = F.relu(self.conv2(x))
        x = F.max_pool2d(z2, 2, 2)
        x = x.view(-1, 16*5*5)
        z3 = F.relu(self.fc1(x))
        a4 = self.fc2(z3)
        return F.log_softmax(a4, dim=1)

    def feature_conv1(self, x):
        return F.relu(self.conv1(x))

    def feature_conv2(self, x):
        z1 = F.relu(self.conv1(x))
        x = F.max_pool2d(z1, 2, 2)
        return F.relu(self.conv2(x))
