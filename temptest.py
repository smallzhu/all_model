import torch

from timm import create_model

import argparse

parser = argparse.ArgumentParser(description="model test")
parser.add_argument('--model', default='resnet50', type=str, metavar='MODEL',
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
    bn_eps=args.bn_eps)

print(model)
input = torch.randn([3,3,384,384])
# model.load_state_dict(torch.load('/45TB/dc/parking/paper_checkpoint/vit_base_patch16_384/1/10_09_01_04_acc9713.pth'))
output = model(input)

print(output)

# #dtys
# import os
# import shutil
# import torch
#
# import torch
# _input = torch.randn(20, 3, 384, 384)
# lstm = torch.nn.LSTM(input_size=384, hidden_size=768, bidirectional=True, batch_first=True)
# linear = torch.nn.Linear(in_features=768, out_features=384)
# # layer = torch.nn.Conv2d(3,3,kernel_size=(3,3), stride=(2,2), padding=(1,1))
# out_1 = lstm(_input)
# out_2 = linear(out_1)
# print(out_2)
# import os
#
# print(len(os.listdir("/45TB/dc/parking/dataset_split_rebuild9_28/train/ab")))
# print(len(os.listdir("/45TB/dc/parking/data_rebuild_11_5/valid/out")))
# import shutil
#
# from tqdm import tqdm
#
# src_path = "/45TB/dc/parking/unlabeld_dst/is"
# dst = "/45TB/dc/parking/data_rebuild_11_17/valid/ab/"
# files = os.listdir(src_path)
# for file in tqdm(files):
#     file_path = os.path.join(src_path, file)
#     shutil.copy(file_path, dst)
# print([1,2])