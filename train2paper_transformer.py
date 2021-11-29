
#这个用于训练二分类的

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

from timm import create_model, loss

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

import argparse


def train(model, criterion, optimizer, num_epochs=100000):
    best_valid_acc = 0
    best_train_acc = 0
    for epoch in range(num_epochs):

        logging.info('Epoch {}/{}'.format(epoch, num_epochs-1))
        logging.info('-' * 10)
        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
            running_loss = 0.0
            running_corrects = 0

            #Iterate over data
            for inputs, labels in tqdm(dataloaders[phase]):
                inputs = inputs.to(devcie)
                # if phase == "train":
                #     vis.images(inputs, win='input_train')
                # else:
                #     vis.images(inputs, win='input_valid')
                labels = labels.to(devcie)

                optimizer.zero_grad()
                with torch.set_grad_enabled(phase=='train'):
                    output = model(inputs)
                    _, preds = torch.max(output, 1)
                    loss = criterion(output, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss /dataset_sizes[phase]
            epoch_acc = running_corrects.double()/ dataset_sizes[phase]
            if phase == 'train':
                if epoch_acc > best_train_acc:
                    best_train_acc = epoch_acc
            if phase == 'valid':
                if epoch_acc > best_valid_acc:
                    best_valid_acc = epoch_acc
                    torch.save(model.state_dict(),os.path.join(save_root, '{}_{}_acc{}.pth'.format(args.num_classes, time.strftime("%m_%d_%H_%M", time.localtime()), int(best_valid_acc*10000))))
            logging.info('{} loss : {:.4f} Acc:{:.4f}'.format(phase, epoch_loss, epoch_acc))

            #数据保存入csv文件
            record_rows = [{'epoch': epoch, 'phase':phase, 'epoch_loss':epoch_loss, 'epoch_acc':epoch_acc, 'local_time': time.strftime("%m_%d_%H_%M", time.localtime())}]
            with open(os.path.join(save_root, args.record_file).format(args.model), 'a+', newline='')as f:
                f_csv = csv.DictWriter(f,record_headers)
                f_csv.writerows(record_rows)

        # print("*"*60)
        print("*"*100)
        logging.info('best train acc: {}    best valid acc: {}'.format(best_train_acc, best_valid_acc))
        print("*"*100)
        # print("*"*60)
        # adjust_lr_scheduler.step()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="model test")
    parser.add_argument('--model', default='vgg16', type=str, metavar='MODEL',
                        help='Name of model to train (default: "resnet50"')
    parser.add_argument('--pretrained', action='store_true', default=True,
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
    parser.add_argument('--record-file', default="record.csv")
    parser.add_argument('--batch-size', type=int, default=10)
    parser.add_argument('--resize',type=int, default=(570, 336))
    parser.add_argument('--save-name', type=str, default="__")

    parser.add_argument('--device', type=str, default="0",help="device")
    parser.add_argument('--lr', type=float, default=0.00001)
    args = parser.parse_args()

    print(args.device)
    os.environ['CUDA_VISIBLE_DEVICES'] = "0,1"
    devcie = torch.device("cuda")

    batch_size = 10
    save_root = "/21TB/dc/parking/checkpoint/{}/{}".format(args.save_name,args.model)
    if not os.path.exists(save_root):
        os.makedirs(save_root)
    record_headers = ['epoch', 'phase', 'epoch_loss', 'epoch_acc', 'local_time']
    with open(os.path.join(save_root, args.record_file),'a+',newline='')as f:
        f_csv = csv.DictWriter(f, record_headers)
        f_csv.writeheader()

    # vis = visdom.Visdom(env='train_2_b4')

    trans = transforms.Compose([
        transforms.RandomChoice(
            [transforms.Resize((args.resize, args.resize)),
             transforms.RandomResizedCrop((args.resize, args.resize), scale=(0.6, 1))]
        ),
        transforms.RandomApply(
            [transforms.ColorJitter(brightness=0.5, contrast=0.3, saturation=0.3)],p=0.3
        ),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomApply(
            [transforms.RandomAffine(degrees=30, translate=(0.1, 0.1), shear=(20,20,20,20))], p=0.3),  # 以0.2的概率应用旋转
        # transforms.RandomApply(
        #     [transforms.RandomRotation(20)], p=0.3
        # ),
        transforms.ToTensor(),
        transforms.RandomErasing(p=0.3, scale=(0.02, 0.33),
                                 ratio=(0.3, 3.3), value="random",
                                 inplace=False),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5])
    ])

    data_dir = "/home/dc/parking/dataset/dataset_split_rebuild9_28/"
    parking_datasets = {x: dataloader.dataset(img_height=args.resize, img_width=args.resize, images_path=data_dir, num_classes=args.num_classes, train_valid=x, trans=trans)
                        for x in ["train", "valid"]}

    dataloaders = {x: torch.utils.data.DataLoader(parking_datasets[x], batch_size=args.batch_size, shuffle=True, num_workers=8)
                   for x in ['train', 'valid']}


    dataset_sizes = {x: len(parking_datasets[x]) for x in ['train', 'valid']}
    print(dataset_sizes['train'])
    print(dataset_sizes['valid'])


    # model = EfficientNet.from_pretrained('efficientnet-b0', num_classes=num_classes, image_size=(img_height, img_width))

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
    print(model)
    model = nn.DataParallel(model)

    if not args.pretrained:
        model.load_state_dict(torch.load('/21TB/dc/parking/checkpoint/lstm_layer/lstm_vit_11_17_14_00_acc9226.pth'))
    logging.info("model loaded")

    model = model.to(devcie)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    # adjust_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
    # criterion = SmoothCrossEntropy()
    criterion = loss.cross_entropy.LabelSmoothingCrossEntropy()

    train(model, criterion, optimizer)




