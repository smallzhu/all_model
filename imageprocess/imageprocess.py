import cv2
import os
import numpy as np
import shutil
from tqdm import tqdm

root_path = "/home/dc/parking/dataset/dataset_split_rebuild9_28/orig_ab"
save_path = "/home/dc/parking/dataset/dataset_split_rebuild9_28/train/ab"

files = os.listdir(root_path)

for file in tqdm(files):
    file_path = os.path.join(root_path, file)
    img = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), 1)
    #水平翻转
    imageflip = cv2.flip(img, 1)
    save_file_path = os.path.join(save_path, "h_"+file)
    cv2.imencode('.jpg', imageflip)[1].tofile(save_file_path)
    #垂直翻转
    imageflip = cv2.flip(img, 0)
    save_file_path = os.path.join(save_path, "v_"+file)
    cv2.imencode('.jpg', imageflip)[1].tofile(save_file_path)
    #复制原图片
    shutil.copy(file_path, save_path)
    #     shutil.copy(file_path, save_path)
print(len(os.listdir(root_path)))
print(len(os.listdir(save_path)))


