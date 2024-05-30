import torchvision
from torch import nn

# dataset = torchvision.datasets.ImageNet("../imagenet", split="train",download=True,
#                                         transform=torchvision.transforms.ToTensor())

vgg16_false = torchvision.models.vgg16(pretrained=False)
vgg16_true = torchvision.models.vgg16(pretrained=True)

print(vgg16_true)

train_data = torchvision.datasets.CIFAR10("../cifar10", train=True,transform=torchvision.transforms.ToTensor(),
                            download=True)

vgg16_true.classifier.add_module('add_linear', nn.Linear(1000, 10))
print(vgg16_true)

print(vgg16_false)
vgg16_false.classifier[6] = nn.Linear(4096, 10)
print(vgg16_false)