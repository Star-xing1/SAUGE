# Code used for data processing
from torch.utils import data
import os
from os.path import join
import cv2 as cv
import numpy as np
import torch
import imageio
import torchvision.transforms as transforms
from PIL import Image
import scipy.io
from torch.distributions import Normal, Independent

class BSDS_Loader(data.Dataset):
    """
    Dataloader BSDS500
    """
    def __init__(self, root='data/HED-BSDS_PASCAL', split='train', transform=False, thresold=0.2):
        self.root = root
        self.split = split
        self.transform = transform
        self.thresold = thresold
        self.num_labels = 3
        if self.split == 'train':
            #self.filelist = join(self.root, 'img_train_pair_flipped_all_labels_rotated.txt')
            self.filelist = join(self.root, 'train_val_all.lst')

        elif self.split == 'test':
            self.filelist = join(self.root, 'test.lst')
        else:
            raise ValueError("Invalid split type!")
        with open(self.filelist, 'r') as f:
            self.filelist = f.readlines()

    def __len__(self):
        return len(self.filelist)
    
    def __getitem__(self, index):
        if self.split == "train":
            img_lb_file = self.filelist[index].strip("\n").split(" ")
            img_file=img_lb_file[0]
            label_list=[]
            label_postive_num_list = []
            for i_label in range(1,len(img_lb_file)):
                lb = scipy.io.loadmat(join(self.root,img_lb_file[i_label]))
                lb=np.asarray(lb['edge_gt'])
                label = torch.from_numpy(lb)
                label = label[1:label.size(0), 1:label.size(1)]
                label = label.float()
                label_postive_num_list.append(torch.sum(label == 1.0).item())
                label_list.append(label.unsqueeze(0))
            sorted_index = sorted(range(len(label_postive_num_list)), key=lambda k: label_postive_num_list[k])
            labels=torch.cat(label_list,0)
            lb_mean=labels.mean(dim=0).unsqueeze(0)
            lb_std=labels.std(dim=0).unsqueeze(0)

            
        else:
            img_file = self.filelist[index].rstrip()

        img = imageio.imread(join(self.root,img_file))
        img = transforms.ToTensor()(img)
        img = img[:, 1:img.size(1), 1:img.size(2)]
        img = img.float()

        if self.split == "train":
            lb_dist= Independent(Normal(loc=lb_mean, scale=lb_std+0.001), 1)
            label = lb_dist.rsample()
            one = torch.ones_like(label)
            zero = torch.zeros_like(label)
            label_thresold = torch.where(label >= self.thresold, one, zero).float()
            lb_sorted = []
            interval = len(img_lb_file) // 2
            # Coarse
            lb_sorted.append(label_list[sorted_index[0]])
            # Middle
            lb_sorted.append(((label_list[sorted_index[0]] + label_list[sorted_index[interval]]) > 0).float() )
            # Fine
            lb_sorted.append(((label_list[sorted_index[0]] + label_list[sorted_index[interval]] + label_list[sorted_index[-1]]) > 0).float())

            lb_sorted = torch.cat(lb_sorted, dim=0)
            return img, lb_sorted, label_thresold, label

        else:
            return img


class Multicue_Loader(data.Dataset):
    """
    Dataloader Multicue
    """
    def __init__(self, type, fold, root='data/multicue', split='train', transform=False, thresold=0.1, crop=300):
        self.root = root
        self.split = split
        self.transform = transform
        self.thresold = thresold
        self.crop = crop
        if self.split == 'train':
            self.filelist = join(self.root, 'train_{}_all_{}.lst'.format(type,fold))
            
        elif self.split == 'test':
            self.filelist = join(self.root, 'test_{}_all_{}.lst'.format(type,fold))
        else:
            raise ValueError("Invalid split type!")
        with open(self.filelist, 'r') as f:
            self.filelist = f.readlines()

    def crop_500(self, img, gt_list):
        w, h = img.size
        random_w = np.random.randint(0, w-512)
        random_h = np.random.randint(0, h-512)

        box = (random_w, random_h, random_w+512, random_h+512)
        img = img.crop(box)
        for index in range(len(gt_list)):
            gt_list[index] = gt_list[index].crop(box) 
            gt_list[index] = torch.from_numpy(np.array(gt_list[index]))
            gt_list[index] = gt_list[index].unsqueeze(0)
            gt_list[index] = gt_list[index].float()

        img = transforms.ToTensor()(img)
        img = img.float()

        return img,gt_list
    
    def crop_320(self, img, gt_list):
        w, h = img.size
        random_w = np.random.randint(0, w-320)
        random_h = np.random.randint(0, h-320)

        box = (random_w, random_h, random_w+320, random_h+320)
        img = img.crop(box)
        for index in range(len(gt_list)):
            gt_list[index] = gt_list[index].crop(box) 
            gt_list[index] = torch.from_numpy(np.array(gt_list[index]))
            gt_list[index] = gt_list[index].unsqueeze(0)
            gt_list[index] = gt_list[index].float()

        img = transforms.ToTensor()(img)
        img = img.float()
        img = torch.nan_to_num(img, nan=0)

        return img,gt_list
    
    def no_crop(self, img, gt_list):
        for index in range(len(gt_list)):
            gt_list[index] = torch.from_numpy(np.array(gt_list[index]))
            gt_list[index] = gt_list[index].unsqueeze(0)
            gt_list[index] = gt_list[index].float()
        img = transforms.ToTensor()(img)
        img = img.float()

        return img,gt_list

    def __len__(self):
        return len(self.filelist)
    
    def __getitem__(self, index):
        if self.split == "train":
            img_lb_file = self.filelist[index].strip("\n").split(" ")
            img_file=img_lb_file[0]
            label_list=[]
            for i_label in range(1,len(img_lb_file)):
                lb = scipy.io.loadmat(join(self.root,img_lb_file[i_label]))
                lb=np.asarray(lb['edge_gt'])
                lb=Image.fromarray(lb)
                label_list.append(lb)
        else:
            img_file = self.filelist[index].strip("\n").split(" ")[0]

        # img = imageio.imread(join(self.root,img_file))
        img=Image.open(join(self.root,img_file)).convert('RGB')
        
        if self.split == "train":
            if self.crop == 300:
                img,label_list=self.crop_320(img,label_list)
            else:
                img,label_list=self.crop_500(img,label_list)
            # img,label_list=self.crop_320(img,label_list)
            labels=torch.cat(label_list,0)
            lb_mean=labels.mean(dim=0).unsqueeze(0)
            lb_std=labels.std(dim=0).unsqueeze(0)


            lb_dist= Independent(Normal(loc=lb_mean, scale=lb_std+0.001), 1)
            label = lb_dist.rsample()
            one = torch.ones_like(label)
            zero = torch.zeros_like(label)
            # edge is 0.1, bound is 0.3
            label_thresold = torch.where(label >= self.thresold, one, zero).float()
            lb_sum = torch.where(lb_mean >= 0.001, one, zero).float()
            # lb_index=random.randint(1,len(label_list))-1
            # label=label_list[lb_index]
            
            lb_sorted = []
            interval = len(label_list) // 2
            label_postive_num_list = []
            for lb in label_list:
                label_postive_num_list.append(torch.sum(lb == 1.0).item())
            sorted_index = sorted(range(len(label_postive_num_list)), key=lambda k: label_postive_num_list[k])

            # Coarse
            lb_sorted.append(label_list[sorted_index[0]])
            # Medium
            lb_sorted.append(((label_list[sorted_index[0]] + label_list[sorted_index[interval]]) > 0).float() )
            # Fine
            lb_sorted.append(((label_list[sorted_index[0]] + label_list[sorted_index[interval]] + label_list[sorted_index[-1]]) > 0).float())
            

            lb_sorted = torch.cat(lb_sorted, dim=0)

            return img, lb_sorted, label_thresold, label
            #,init.permute(2,0,1)
        else:
            img = transforms.ToTensor()(img)
            img=img.float()
            return img
            #,init.permute(2,0,1)


class NYUD_Loader(data.Dataset):
    def __init__(self, root='data/', split='test', transform=False, threshold=0.4, setting=['image']):
        self.root = root
        self.split = split
        if self.split == 'train':
            self.filelist = os.path.join(
                    self.root, '%s-train_da.lst' % (setting[0]))
        elif self.split == 'test':
            self.filelist = os.path.join(
                    self.root, '%s-test.lst' % (setting[0]))
        else:
            raise ValueError("Invalid split type!")
        with open(self.filelist, 'r') as f:
            self.filelist = f.readlines()

    def __len__(self):
        return len(self.filelist)
    
    def __getitem__(self, index):
        scale = 1.0
        if self.split == "train":
            img_file, lb_file, scale = self.filelist[index].split()
            img_file = img_file.strip()
            lb_file = lb_file.strip()
            scale = float(scale.strip())
            pil_image = Image.open(os.path.join(self.root, lb_file))
            if scale < 0.99:
                W = int(scale * pil_image.width)
                H = int(scale * pil_image.height)
                pil_image = pil_image.resize((W, H))
            lb = np.array(pil_image, dtype=np.float32)
            if lb.ndim == 3:
                lb = np.squeeze(lb[:, :, 0])
            assert lb.ndim == 2
            threshold = self.threshold
            lb = lb[np.newaxis, :, :]
            lb[lb == 0] = 0
            lb[np.logical_and(lb>0, lb<threshold)] = 2
            lb[lb >= threshold] = 1
            
        else:
            img_file = self.filelist[index].rstrip()

        img = imageio.imread(join(self.root,img_file))
        img = transforms.ToTensor()(img)
        img = img.float()

        if self.split == "train":
            return img, lb
        else:
            return img