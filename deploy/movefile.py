import argparse
import os

import shutil

import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from torchvision.transforms import ToTensor, Resize, Compose, Normalize
from tqdm import tqdm

from timm import create_model
import torch

if __name__ == "__main__":

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

    parser.add_argument('--batch-size', type=int, default=10)
    parser.add_argument('--resize',type=int, default=(570, 336))

    parser.add_argument('--device', type=str, default="0",help="device")
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = "2,3,4,5"

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
        bn_eps=args.bn_eps,
)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    weights = r"/45TB/dc/parking/paper_checkpoint/vit_base_patch16_384/1/10_09_01_04_acc9713.pth"
    model.load_state_dict(torch.load(weights))

    model = model.to(device)
    model.eval()
    # image_path = r"E:\parking\EfficientNet-PyTorch\images\20210605112956193_20041.jpg"
    # class2 = Classify(model=model, num_classes=3, image_size=(384, 384), device = device, weights=weights)
    folders = ["ab", "in", "out"]
    classes = ["in", "out", "ab"]
    transform = Compose([Resize((384, 384)), ToTensor(), Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )])

    root_path = r"/45TB/dc/parking/unlabled"
    dst_path_is = r"/45TB/dc/parking/unlabeld_dst/is"
    dst_path_not = r"/45TB/dc/parking/unlabeld_dst/no"
    num_files = 0.0
    correct = 0.0
    folder_path = os.path.join(root_path)
    files = os.listdir(folder_path)
    print(num_files)
    bar = tqdm(files)
    for file in bar:
        num_files+=1
        file_path = os.path.join(folder_path, file)
        image_input = Image.open(file_path)
        image_input = transform(image_input)
        image_input = image_input.unsqueeze(0)
        image_input = image_input.to(device)

        output = model(image_input)
        preds_index, preds = torch.max(output, 1)
        if classes[preds.data] == "ab":
            shutil.copy(file_path, dst_path_is)

        correct_dev = correct/num_files

        tqdm.write(f" pred: {classes[preds.data]}, "
                   f"acc {correct_dev:.4f} correct {correct} numfiles {num_files}"
                   f" {file}")

            # bar.set_description(f"{folder}, {classes[preds.data]}, {correct_dev:.4f}, {file}")

    # for file in tqdm(files):
    #     file_path = os.path.join(files_path,file)
    #     pred = class2.detect(file_path)
    #
    #     if pred[0] == 'in':
    #         shutil.move(file_path, r"E:\parking\parking_data\santu_dst\in")
    #     elif pred[0] == "out":
    #         shutil.move(file_path, r"E:\parking\parking_data\santu_dst\out")
    #     elif pred[0] == 'ab':
    #         shutil.move(file_path, r"E:\parking\parking_data\santu_dst\ab")


