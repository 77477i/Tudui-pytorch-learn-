from torch.utils.tensorboard import SummaryWriter
import numpy as np
from PIL import Image

writer = SummaryWriter("../logs")
image_path = "../data/train/ants_image/0013035.jpg"
img_PIL = Image.open(image_path) # class 'PIL.JpegImagePlugin.JpegImageFile'
img_array = np.array(img_PIL) # <class 'numpy.ndarray'>对图片的格式进行转换'
print(type(img_array))
print(img_array.shape)


writer.add_image("train", img_array, 2, dataformats='HWC')
#writer.add_scalar()


for i in range(100):
    writer.add_scalar("y = 3x", 3 * i, i)

writer.close()