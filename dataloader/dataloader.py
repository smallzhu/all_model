import os
import torch.utils.data as data
import numpy as np

from torchvision import transforms
from PIL import Image

def get_filepath(dir_root):
    file_paths = []
    for root, dirs, files in os.walk(dir_root):
        for file in files:
            file_paths.append(os.path.join(root, file))
    return file_paths

class dataset(data.Dataset):
    def __init__(self,img_height, img_width, images_path, num_classes, train_valid, trans):
        self.img_height = img_height
        self.img_width = img_width
        self.num_classes = num_classes
        self.train_vlid = train_valid
        if train_valid == "train":
            images_path = os.path.join(images_path, "train")
        else:
            images_path = os.path.join(images_path, "valid")

        self.data = get_filepath(images_path)
        if trans:
            self.data_transform = trans

        self.valid_transform = transforms.Compose([
            transforms.Resize((self.img_height, self.img_width)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                 std=[0.5, 0.5, 0.5])
        ])
    def __getitem__(self, item):
        img_path = self.data[item]
        if self.num_classes == 2: #分类数为2将 in out 文件夹作为一个类，normal为另外一个类
            if img_path.split('/')[-2] == 'in':
                label = 0
            elif img_path.split('/')[-2] == 'out':
                label = 0
            elif img_path.split('/')[-2] == 'ab':
                label = 1
        elif self.num_classes == 3:#分类数为3时，将in out abnormal分别视为0，1，2三个类
            if img_path.split('/')[-2] == 'in':
                label = 0
            elif img_path.split('/')[-2] == 'out':
                label = 1
            elif img_path.split('/')[-2] == 'ab':
                label = 2

        img = Image.open(img_path)
        if img.size[0] > img.size[1]:
            img = self.imageprocess(img)

        if self.train_vlid == 'train':
            img = self.data_transform(img)
        else:
            img = self.valid_transform(img)   #减少验证集的图片处理更贴合实际图片传入

        return img, label

    def __len__(self):
        return len(self.data)

    def histeq(self,imarr):
        hist, bins = np.histogram(imarr, 255)
        cdf = np.cumsum(hist)
        cdf = 255 * (cdf/cdf[-1])
        res = np.interp(imarr.flatten(), bins[:-1], cdf)
        res = res.reshape(imarr.shape)
        return res, hist

    def imageprocess(self, image):
        image = image.crop((0, 0, 2688, 1520))
        image_1 = image.crop((0, 0, 1344, 760))
        image_2 = image.crop((1344, 0, 2688, 760))
        image_3 = image.crop((0, 760, 1344, 1520))


        target = Image.new('RGB', (1344, 2280))
        target.paste(image_1, (0, 0))
        target.paste(image_2, (0, 760))
        target.paste(image_3, (0, 1520))

        return target

