import torch
import torchvision
from torch import nn
from torch.nn import Conv2d
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10("cifar10", train=False,
                                       transform=torchvision.transforms.ToTensor(),
                                       download=True)
dataloder = DataLoader(dataset, batch_size=64)

class Kiki(nn.Module):
    def __init__(self):
        super(Kiki, self).__init__()
        self.conv1 = Conv2d(3, 6, 3, stride=1, padding=0)

    def forward(self, x):
        x = self.conv1(x)
        return x

kiki = Kiki()

writer = SummaryWriter("../logs")

step = 0
for data in dataloder:
    imgs, targets = data
    output = kiki(imgs)
    print(imgs.shape)
    print(output.shape)
    # torch.Size([64, 3, 32, 32])
    writer.add_images("input", imgs, step)
    # torch.Size([64, 6, 30, 30])

    output = torch.reshape(output, (-1, 3, 30, 30)) # 第一个值不知道多少的时候，直接写-1，会自动根据后面的进行计算
    writer.add_images("output", output, step)
    step += 1




