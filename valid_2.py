# 这个用于训练二分类的

import logging
import os
import time

import PIL
import csv
import torch

from torch import optim, nn
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm

from dataloader import dataloader
from torch.optim import lr_scheduler

from util.util import AverageMeter
from timm import create_model, loss

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

import argparse


def valid(model):
    model.eval()
    running_corrects = AverageMeter()
    # Iterate over data
    bar = tqdm(dataloaders)
    running_loss = 0
    correct = 0
    num_image = 0
    for inputs, labels in bar:
        inputs = inputs.to(devcie)

        labels = labels.to(devcie)

        with torch.set_grad_enabled(True):
            output = model(inputs)
            _, preds = torch.max(output, 1)
            # loss = criterion(output, labels)
        # running_loss += loss.item() * inputs.size(0)
        batch_corecct = 0
        for pred, label in zip(preds,labels.data):
            if pred == 0 or pred == 1:
                if label == 0 or label == 1:
                    batch_corecct += 1
            if pred == 2 and label == 2:
                batch_corecct += 1

        correct += batch_corecct
        num_image += args.batch_size
        # running_corrects.update(int(x), args.batch_size)
        bar.set_description(f"{correct/num_image}")
        # bar.set_description(f"{running_corrects.avg}   {running_loss}")

    # epoch_acc = running_corrects.double()/ dataset_sizes

    # logging.info('Acc:{:.4f}'.format(epoch_acc))


if __name__ == '__main__':
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
    parser.add_argument('--resize', type=int, default=384)

    parser.add_argument('--device', type=str, default="0", help="device")
    args = parser.parse_args()

    # print(args.device)
    os.environ['CUDA_VISIBLE_DEVICES'] = "1,2,3,4,5"
    devcie = torch.device("cuda")

    batch_size = 6
    lr = 0.00001
    save_root = "/45TB/dc/parking/paper_checkpoint/{}".format(args.model)

    data_dir = "/45TB/dc/parking/dataset_split_rebuild9_28"
    parking_datasets = dataloader.dataset(img_height=args.resize, img_width=args.resize, images_path=data_dir,
                                          num_classes=args.num_classes, train_valid="valid", trans=0)

    dataloaders = torch.utils.data.DataLoader(parking_datasets, batch_size=args.batch_size, shuffle=True, num_workers=8)

    dataset_sizes = len(parking_datasets)
    print(dataset_sizes)

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

    # print(model)
    model.load_state_dict(
        torch.load('/45TB/dc/parking/paper_checkpoint/vit_base_patch16_384/1/10_09_01_04_acc9713.pth'))
    model = nn.DataParallel(model)
    logging.info("model loaded")

    model = model.to(devcie)

    # criterion = loss.IB_FocalLoss()

    valid(model)




