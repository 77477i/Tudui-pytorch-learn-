import torchvision
from torch import nn
from torch.nn import Sequential, Conv2d, MaxPool2d, Flatten, Linear
from torch.utils.data import DataLoader

dataset = torchvision.datasets.CIFAR10("../cifar10", train=False, transform=torchvision.transforms.ToTensor(),
                                       download=True)

dataloader = DataLoader(dataset, batch_size=1)

class Kiki(nn.Module):
    def __init__(self):
        super(Kiki, self).__init__()
        self.model1 = Sequential(
            Conv2d(3,32,5,padding=2),
            MaxPool2d(2),
            Conv2d(32, 32, 5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 64, 5, padding=2),
            MaxPool2d(2),
            Flatten(),
            Linear(1024, 64),
            Linear(64, 10)
        )


    def forward(self, x):
        x = self.model1(x)
        return x

loss = nn.CrossEntropyLoss()
kiki = Kiki()
for data in dataloader:
    imgs,tagets = data
    outputs = kiki(imgs)
    result_loss = loss(outputs, tagets)
    result_loss.backward() # 对loss后的进行处理
    print("ok")