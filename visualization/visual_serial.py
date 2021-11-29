import logging

import torch
import torch.nn as nn
from visualization.core import *
from visualization.core.utils import image_net_postprocessing
from PIL import Image
from efficientnet_pytorch import EfficientNet

from torchvision.models import alexnet, vgg16, resnet18, resnet152
from torchvision.transforms import ToTensor, Resize, Compose

import matplotlib.pyplot as plt
from visualization.core.utils import image_net_postprocessing, image_net_preprocessing

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# create a model
model = EfficientNet.from_name('efficientnet-b0',num_classes=2, image_size=(570, 336))
model = nn.DataParallel(model)
model.load_state_dict(torch.load(r"E:\parking\EfficientNet-PyTorch\checkpoint\b0_07_28_08_17_acc9695.pth"))
logging.info("model loaded")

model = model.to(device)
model.eval()

classss = ["normal", "abnormal"]


def imshow(tensor):
    tensor = tensor.squeeze()
    if len(tensor.shape) > 2: tensor = tensor.permute(1, 2, 0)
    img = tensor.cpu().numpy()
    plt.imshow(img, cmap='gray')
    plt.title('{}'.format(classss[preds.data]))
    plt.show()

while True:
    try:
        image_path = input("input image path \n")
        # print(image_path)
    except Exception as e:
        print("invalid image path! retry")
        continue
    else:
        # print(model)
        cat = Image.open(image_path)
        # resize the image and make it a tensor
        input_image = Compose([Resize((570, 336)), ToTensor(), image_net_preprocessing])(cat)
        # add 1 dim for batch
        input_image = input_image.unsqueeze(0)
        input_image = input_image.to(device)
        # call mirror with the input and the model
        layers = list(model.children())

        output = model(input_image)
        # print(output)
        _, preds = torch.max(output, 1)
        print(classss[preds.data])

        vis = GradCam(model.to(device), device)
        img = vis(input_image.to(device), None,
                  target_class=None,
                  postprocessing=image_net_postprocessing,
                  guide=False)

        with torch.no_grad():
            imshow(img[0])


