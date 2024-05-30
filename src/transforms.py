from torch.utils.tensorboard import SummaryWriter
from torchvision import  transforms
from PIL import Image

# 通过transforms.totensor去解决两个问题
# 1、transforms如何使用（python）
# 2、tensor的数据类型与普通的数据类型有什么区别，有什么用
img_path = r"../data/train/ants_image/0013035.jpg"
img = Image.open(img_path)
print(img)

writer = SummaryWriter("../logs")

tensor_trans = transforms.ToTensor()
tensor_img = tensor_trans(img)
print(tensor_img)

writer.add_image("Tensor_img", tensor_img)
writer.close()

