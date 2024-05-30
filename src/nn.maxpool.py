import torch
import torchvision
from torch import nn
from torch.nn import MaxPool2d
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10("cifar10",train=False, download=True,
                                       transform=torchvision.transforms.ToTensor())
dataloader = DataLoader(dataset, 64)

class Kiki(nn.Module):
    # 初始化
    def __init__(self):
        super(Kiki, self).__init__()
        self.maxpool1 = MaxPool2d(kernel_size=3, ceil_mode=False)

    # 前向
    def forward(self, input):
        output = self.maxpool1(input)
        return output

kiki = Kiki()

writer = SummaryWriter("../logs_maxpool")
step = 0

for data in dataloader:
    imgs, targets = data
    writer.add_images("input", imgs, step)
    output =kiki(imgs)
    writer.add_images("output", output, step)
    step += 1

writer.close()
