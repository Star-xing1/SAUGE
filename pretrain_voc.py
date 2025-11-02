"""
The code for pre-trainning SAUGE on VOC. 
The framework of trainning code is based on https://github.com/ZhouCX117/UAED_MuGE.
"""
import torch
from torch import nn
train_root="./"
import os, sys
from statistics import mode
sys.path.append(train_root)

import numpy as np
from PIL import Image
import argparse
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import matplotlib
matplotlib.use('Agg')

from data.data_loader_pretrain_voc import VOC_RCFLoader
MODEL_NAME="model.sauge_vitb"
import importlib
Model = importlib.import_module(MODEL_NAME)

from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.backends.cudnn as cudnn
from utils import Logger, Averagvalue, save_checkpoint
from os.path import join, split, isdir, splitext, split, abspath, dirname
import scipy.io as io
from shutil import copyfile
import random
import numpy
from torch.autograd import Variable
import ssl
import cv2
ssl._create_default_https_context = ssl._create_unverified_context
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument('--batch_size', default=3, type=int, metavar='BT',
                    help='batch size')
# =============== optimizer
parser.add_argument('--LR', '--learning_rate', default=0.0001, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--weight_decay', '--wd', default=0.0005, type=float,
                    metavar='W', help='default weight decay')
parser.add_argument('--stepsize', default=6, type=int, 
                    metavar='SS', help='learning rate step size')
parser.add_argument('--maxepoch', default=20, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--print_freq', '-p', default=200, type=int,
                    metavar='N', help='print frequency (default: 50)')
parser.add_argument('--gpus', default='4,6,7', type=str,
                    help='GPU ID')
parser.add_argument('--tmp', help='tmp folder', default='logs/voc_pretrain/vitb/')
parser.add_argument('--dataset', help='root folder of dataset', default='./data_file/PASCAL')
parser.add_argument('--itersize', default=1, type=int,
                    metavar='IS', help='iter size')
parser.add_argument('--sam_type', default="vit_h", type=str,help='type of sam model')
parser.add_argument('--sam_ckpt', default="sam_vit_h_4b8939.pth", type=str,help='ckpt path of sam model')

parser.add_argument("--local_rank", default=os.getenv('LOCAL_RANK', -1), type=int)
args = parser.parse_args()


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
THIS_DIR = abspath(dirname(__file__))
TMP_DIR = join(THIS_DIR, args.tmp+"{}".format(MODEL_NAME[6:]))
if not isdir(TMP_DIR):
  os.makedirs(TMP_DIR)

file_name=os.path.basename(__file__)
copyfile(join(train_root,MODEL_NAME[:5],MODEL_NAME[6:]+".py"),join(TMP_DIR,MODEL_NAME[6:]+".py"))
copyfile(join(train_root, file_name),join(TMP_DIR,file_name))
def init_seeds(seed=0, cuda_deterministic=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda_deterministic:  # slower, more reproducible
       cudnn.deterministic = True
       cudnn.benchmark = False
    else:  # faster, less reproducible
       cudnn.deterministic = False
       cudnn.benchmark = True

def reduce_tensor(tensor: torch.Tensor):
    rt = tensor.clone()
    torch.distributed.all_reduce(rt, op=torch.distributed.reduce_op.SUM)
    rt /= torch.distributed.get_world_size()
    return rt

def cross_entropy_loss_RCF(prediction, labelef, label_prob=None):
    label = labelef.long()
    mask = label.float()
    num_positive = torch.sum((mask==1).float()).float()
    num_negative = torch.sum((mask==0).float()).float()
    num_two=torch.sum((mask==2).float()).float()
    assert num_negative+num_positive+num_two==label.shape[0]*label.shape[1]*label.shape[2]*label.shape[3]
    assert num_two==0
    mask[mask == 1] = 1.0 * num_negative / (num_positive + num_negative)
    mask[mask == 0] = 1.1 * num_positive / (num_positive + num_negative)
    mask[mask == 2] = 0
    if label_prob is not None:
        new_mask=mask * torch.exp(label_prob)
    else:
        new_mask=mask
    cost = F.binary_cross_entropy(
                prediction, labelef, weight=new_mask.detach(), reduction='sum')
     
    return cost


def main():
    if args.local_rank != -1:
        torch.cuda.set_device(args.local_rank)
        device=torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method='env://')

    sam = sam_model_registry[args.sam_type](checkpoint=args.sam_ckpt)
    for name, parameter in sam.named_parameters():
        parameter.requires_grad = False
    sam.to(device=device)
    num_gpus = torch.cuda.device_count()
    mask_generator = SamAutomaticMaskGenerator(sam)
    args.cuda = True
    train_dataset = VOC_RCFLoader(root=args.dataset, split= "train")
    train_sampler = DistributedSampler(train_dataset)
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size,
        num_workers=4, drop_last=True, sampler=train_sampler)
    # model
    model=Model.Mymodel(args, sam_generator=mask_generator).to(device)
    if num_gpus > 1:
        model = nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                    output_device=args.local_rank)
        

    log = Logger(join(TMP_DIR, '%s-%d-log.txt' %('Adam',args.LR)))
    sys.stdout = log
    
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.LR, weight_decay=args.weight_decay)
    
    for epoch in range(args.start_epoch, args.maxepoch):
        train_sampler.set_epoch(epoch)
        model.train()
        train(train_loader, model, optimizer,epoch,
            save_dir = join(TMP_DIR, 'epoch-%d-training-record' % epoch), log=log)
        torch.distributed.barrier()


def train(train_loader, model,optimizer,epoch, save_dir, log):
    batch_time = Averagvalue()
    data_time = Averagvalue()
    losses = Averagvalue()
    l1_loss = nn.L1Loss(reduction='none')
    # rank_loss = nn.MarginRankingLoss(reduction='none', margin=1)

    print(epoch,optimizer.state_dict()['param_groups'][0]['lr'])
    end = time.time()
    epoch_loss = []
    counter = 0
    for i, (image, label) in enumerate(train_loader):
        data_time.update(time.time() - end)
        image, label = image.cuda(), label.cuda()
        outputs, final_output, mask_edge, mask_edge_count = model(image)

        counter += 1
        
        ada=(epoch+1)/args.maxepoch
        w_mask = mask_edge
        mask_gt_01 = torch.sum((~w_mask.bool()).float() * torch.logical_xor(mask_edge, label).float())
        mask_gt_10 = torch.sum((w_mask.bool()).float() * torch.logical_xor(mask_edge, label).float())

        hard_label_01 = ((~w_mask.bool()).float() * torch.logical_xor(mask_edge, label).float()) * (mask_gt_10/(mask_gt_01 + mask_gt_10)) * (-1)
        hard_label_10 = ((w_mask.bool()).float() * torch.logical_xor(mask_edge, label).float() * (mask_edge_count/torch.mean(mask_edge_count[mask_edge_count > 0])) * (-1))
        label_prob_cos = None
        loss = None
        final_loss = cross_entropy_loss_RCF(final_output, label, label_prob_cos)
        for index in range(len(outputs)):
            side_weight = None

            if loss is not None:
                loss += cross_entropy_loss_RCF(outputs[index], label, side_weight)
            else:
                loss = cross_entropy_loss_RCF(outputs[index], label, side_weight)
                    
        loss = 0.5 * loss
        loss += final_loss

        reduced_loss = reduce_tensor(loss)
        reduced_final = reduce_tensor(final_loss)
        loss.backward()
        if counter == args.itersize:
            optimizer.step()
            optimizer.zero_grad()
            counter = 0
        losses.update(loss.item(), image.size(0))
        epoch_loss.append(loss.item())
        batch_time.update(time.time() - end)
        end = time.time()
        # display and logging
        if not isdir(save_dir) and args.local_rank == 0:
            os.makedirs(save_dir)
        if i % args.print_freq == 1 and args.local_rank == 0:
            info = 'Epoch: [{0}/{1}][{2}/{3}] '.format(epoch, args.maxepoch, i, len(train_loader)) + \
                   'Time {batch_time.val:.3f} (avg:{batch_time.avg:.3f}) '.format(batch_time=batch_time) + \
                   'Loss {loss.val:f} (avg:{loss.avg:f}) '.format(
                       loss=losses)
            print(info)
            print(reduced_loss.item(), reduced_final.item())
            

            for index in range(len(outputs)):
                torchvision.utils.save_image(1-outputs[index], join(save_dir, "iter-%d-rank-%d.jpg" % (i, index+1)))
            torchvision.utils.save_image(image, join(save_dir, "iter-%d-img.jpg" % (i)))
            torchvision.utils.save_image(1-mask_edge, join(save_dir, "iter-%d-mask_edge.jpg" % (i)))
            torchvision.utils.save_image(1-final_output, join(save_dir, "iter-%d-final.jpg" % (i)))
            torchvision.utils.save_image(1-label, join(save_dir, "iter-%d-label_final.jpg" % (i)))
            log.flush()
        torch.distributed.barrier()
        # save checkpoint
    if args.local_rank == 0:
        save_checkpoint({
            'epoch': epoch,
            'state_dict': model.state_dict(),
                }, filename=join(save_dir, "epoch-%d-checkpoint.pth" % epoch))


if __name__ == '__main__':
    random_seed = 6666
    init_seeds(random_seed)
    main()
   
