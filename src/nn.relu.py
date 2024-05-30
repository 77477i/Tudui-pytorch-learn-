import torch
import torchvision
from timm.layers import Sigmoid
from torch import nn
from torch.nn import ReLU
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

input = torch.tensor([[1, -0.5],
                      [-1, 3]])

input = torch.reshape(input, (-1, 1, 2, 2))
print(input.shape)

dataset = torchvision.datasets.CIFAR10("cifar10", train=False,
                                       transform=torchvision.transforms.ToTensor(),download=True)

dataloader = DataLoader(dataset, batch_size=64)


class Kiki(nn.Module):
    def __init__(self):
        super(Kiki, self).__init__()
        self.relu1 = ReLU()
        self.sigmoid = Sigmoid()

    def forward(self, input):
        output = self.sigmoid(input)
        return output

kiki = Kiki()

writer = SummaryWriter("../logs_sigmoid")
step = 0

for data in dataloader:
    imgs, targets = data
    writer.add_images("input", imgs, global_step=step)
    output = kiki(imgs)
    writer.add_images("output", output, step)
    step += 1

writer.close()




