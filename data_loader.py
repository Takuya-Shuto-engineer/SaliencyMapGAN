import glob
import os
import numpy as np
import cv2
from tqdm import tqdm
import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from constants import *

class DataLoader(object):
    def __init__(self, batch_size = 5):
        # trainディレクトリ以下のpathを全て取得して，ファイル名のみ切り取る．拡張子も消しておく．
        self.list_img = [k.split('/')[-1].split('.')[0]] for k in glob.glob(os.path.join(pathToResizedImageTrain, '*train*'))
        self.batch_size = batch_size
        self.size = len(self.list_img)
        self.cursor = 0
        self.num_batches = self.size / batch_size
    
    def get_batch(self):
        # 現在見ている場所がサイズを超えている場合のリセット処理
        if self.cursor + self.batch_size > self.size:
            self.cursor = 0
            np.random.shuffle(self.list_img)
        
        # 格納用の変数を先に定義しておく
        img = torch.zeros(self.batch_size, 3, 192, 256)
        sal_map = torch.zeros(self.batch_size, 1, 192, 256)
        
        # 0 - 255 から 0 - 1.0に正規化する規定を追加
        to_tensor = transforms.ToTensor()
        
        for i in range(self.batch_size):
            # バッチサイズ分だけリストから取り出す
            curr_file = self.list_img[self.cursor]
            full_img_path = os.path.join(pathToResizedImagesTrain, curr_file + '.png')
            full_map_path = os.path.join(pathToResizedImagesTrain, curr_file + '.png')
            self.cursor += 1
            inputimage = cv2.imread(full_img_path) # (192, 256, 3)
            img[i] = to_tensor(inputimage)
            
            saliencyimage = cv2.imread(full_map_path, 0) # (192, 256)
            saliencyimage = np.expand_dims(saliencyimage, axis = 2) # (192, 256, 1)に変換
            sal_map[i] = to_tensor(saliencyimage)
        
        return (img, sal_map)
            