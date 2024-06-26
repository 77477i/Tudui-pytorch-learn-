import torch
import torchvision
from torch.nn import Linear
from torch import nn
from torch.utils.data import DataLoader

dataset = torchvision.datasets.CIFAR10("cifar10", train=False, transform=torchvision.transforms.ToTensor(),
                                       download=True)

dataloader = DataLoader(dataset, batch_size=64)

class Kiki(nn.Module):
    def __init__(self):
        super(Kiki, self).__init__()
        self.linear1 = Linear(196608, 10)

    def forward(self, input):
        output = self.linear1(input)
        return output

kiki = Kiki()

for data in dataloader:
    imgs, targets = data
    print(imgs.shape)
    output = torch.reshape(imgs, (1, 1, 1, -1))
    print(output.shape)
    output = kiki(output)
    print(output.shape)