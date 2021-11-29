import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from visualization.core import *
from visualization.core.utils import image_net_postprocessing
from PIL import Image
from timm import create_model

from torchvision.models import alexnet, vgg16, resnet18, resnet152
from torchvision.transforms import ToTensor, Resize, Compose

import matplotlib.pyplot as plt
from visualization.core.utils import image_net_postprocessing, image_net_preprocessing

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


parser = argparse.ArgumentParser(description="model test")
parser.add_argument('--model', default='vit_base_patch16_384', type=str, metavar='MODEL',
                    help='Name of model to train (default: "resnet50"')
parser.add_argument('--pretrained', action='store_true', default=False,
                    help='Start with pretrained version of specified network (if avail)')
parser.add_argument('--num-classes', type=int, default=3, metavar='N',
                    help='number of label classes (Model default if None)')
parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                    help='Dropout rate (default: 0.)')
parser.add_argument('--drop-connect', type=float, default=None, metavar='PCT',
                    help='Drop connect rate, DEPRECATED, use drop-path (default: None)')
parser.add_argument('--drop-path', type=float, default=None, metavar='PCT',
                    help='Drop path rate (default: None)')
parser.add_argument('--drop-block', type=float, default=None, metavar='PCT',
                    help='Drop block rate (default: None)')
parser.add_argument('--gp', default=None, type=str, metavar='POOL',
                    help='Global pool type, one of (fast, avg, max, avgmax, avgmaxc). Model default if None.')
parser.add_argument('--bn-tf', action='store_true', default=False,
                    help='Use Tensorflow BatchNorm defaults for models that support it (default: False)')
parser.add_argument('--bn-momentum', type=float, default=None,
                    help='BatchNorm momentum override (if not None)')
parser.add_argument('--bn-eps', type=float, default=None,
                    help='BatchNorm epsilon override (if not None)')
args = parser.parse_args()

model = create_model(
    args.model,
    pretrained=args.pretrained,
    num_classes=args.num_classes,
    drop_rate=args.drop,
    drop_connect_rate=args.drop_connect,  # DEPRECATED, use drop_path
    drop_path_rate=args.drop_path,
    drop_block_rate=args.drop_block,
    global_pool=args.gp,
    bn_tf=args.bn_tf,
    bn_momentum=args.bn_momentum,
    bn_eps=args.bn_eps,)


model.load_state_dict(torch.load(r"/45TB/dc/parking/paper_checkpoint/vit_base_patch16_384/1/10_09_01_04_acc9713.pth"))
model = model.to(device)
print("model loaded")
model.eval()
# print(model)
cat = Image.open(r"/home/zhudechen/parking/all_model/image/20211010085357108_蓝川AD8S00_2_MT-BGDL-01-03071408_img.jpg")
# resize the image and make it a tensor
input = Compose([Resize((384, 384)), ToTensor(), image_net_preprocessing])(cat)
# add 1 dim for batch
input = input.unsqueeze(0)
input = input.to(device)
# call mirror with the input and the model
layers = list(model.children())
# layer = layers[4][2]
# print(layer)

def imshow(tensor):
    tensor = tensor.squeeze()
    if len(tensor.shape) > 2: tensor = tensor.permute(1, 2, 0)
    img = tensor.cpu().numpy()
    plt.imshow(img, cmap='gray')
    plt.title('{}'.format(classss[preds.data]))
    plt.show()

classss = ["in", "out", "abnormal"]
output = model(input)
soft_output = F.softmax(output, 1)
print(output)
print(soft_output)
preds_index, preds = torch.max(output, 1)
print(classss[preds.data], preds_index.data)

vis = GradCam(model.to(device), device)
img = vis(input.to(device), None,
          target_class=None,
          postprocessing=image_net_postprocessing,
          guide=False)

with torch.no_grad():
    imshow(img[0])


