from torch.utils import data
import os
from os.path import join, abspath, splitext, split, isdir, isfile
import numpy as np
import torch
import imageio
import torchvision.transforms as transforms
import scipy.io
from PIL import Image
import random
import cv2
train_root='/data1/fuxing/data_file/BSDS/PASCAL/PASCAL'
test_root='/data1/fuxing/data_file/BSDS'
class VOC_RCFLoader(data.Dataset):
    """
    Dataloader VOC
    """
    def __init__(self, root='data/HED-BSDS_PASCAL', split='train', transform=False):
        self.root = root
        self.split = split
        self.transform = transform
        if self.split == 'train':
            self.filelist = join(train_root, 'train_pair.lst')
            
        elif self.split == 'test':
            self.filelist = join(test_root, 'test.lst')
        else:
            raise ValueError("Invalid split type!")
        with open(self.filelist, 'r') as f:
            self.filelist = f.readlines()
        self.train_transform=transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.CenterCrop(384),
                
            ]
        )
        self.test_transform=transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.ToTensor()
            ]
        )
    def __len__(self):
        return len(self.filelist)
    
    def crop_320(self, img, gt_list):
        w, h = img.size
        if w-320 <=0:
            random_w = 0
        else:
            random_w = np.random.randint(0, w-320)
        if h - 320 <=0:
            random_h = 0
        else:
            random_h = np.random.randint(0, h-320)

        box = (random_w, random_h, random_w+320, random_h+320)
        img = img.crop(box)
        for index in range(len(gt_list)):
            gt_list[index] = gt_list[index].crop(box) 
            gt_list[index] = torch.from_numpy(np.array(gt_list[index]))
            gt_list[index] = gt_list[index].unsqueeze(0)
            gt_list[index] = gt_list[index].float() # if crop 500 + multi-scale to train, the 1280x720 is able to train

        img = transforms.ToTensor()(img)
        img = img.float()

        return img,gt_list
    
    def __getitem__(self, index):
        if self.split == "train":
            img_lb_file = self.filelist[index].strip("\n").split(" ")
            img_file=img_lb_file[0]
            lb_file=img_lb_file[-1]
            img = Image.open(join(train_root, img_file)).convert('RGB')
        else:
            img_file = self.filelist[index].rstrip()
            img = imageio.imread(join(test_root,img_file))
            img=img[1:,1:,:]
            img=self.test_transform(img)
        
        if self.split == "train":
            lb = np.array(Image.open(os.path.join(self.root, lb_file)), dtype=np.float32)
            lb = lb.astype(np.uint8)
            if lb.ndim == 3:
                lb = np.squeeze(lb[:, :, 0])
            assert lb.ndim == 2
            lb_list = [Image.fromarray(lb)]
            img, lb_list = self.crop_320(img, lb_list)
            lb = lb_list[0]
            lb = lb / 255
            lb[lb > 0] = 1  # Binarize the label
            lb = lb.float()
                
            return img, lb
        else:
            return img
