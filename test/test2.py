import torch

output = torch.tensor([[0.1, 0.2],
                      [0.3, 0.4]])

print(output.argmax(1)) # 1表示横向看两行哪个值更大输出哪个
preds = output.argmax(1)
targets = torch.tensor([0, 1])
print(preds == targets)