#!/user/bin/python
# coding=utf-8
import torch
import os, sys
import numpy as np
from PIL import Image
import argparse
import torch
import torch.nn.functional as F
from tqdm import *

from data.data_loader_granularity import BSDS_Loader, Multicue_Loader, NYUD_Loader
MODEL_NAME="model.sauge_vitb"
import importlib
Model = importlib.import_module(MODEL_NAME)

from torch.utils.data import DataLoader
from os.path import join, split, isdir, splitext, split, abspath, dirname
import scipy.io as io
import random
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamAutomaticMaskGeneratorForTest

parser = argparse.ArgumentParser(description='Test')
parser.add_argument('--device', default="2", type=str, help='the GPU ids')
parser.add_argument('--sam_type', default="vit_b", type=str,help='type of sam model')
parser.add_argument('--sam_ckpt', default="/data1/fuxing/segment-anything/ckpt/sam_vit_b_01ec64.pth", type=str,help='ckpt path of sam model')
parser.add_argument('--dataset', help='root folder of dataset', default='/data1/fuxing/UAED/data_file/BSDS')
# parser.add_argument('--dataset', help='root folder of dataset', default='data_path/multicue')
# parser.add_argument('--dataset', help='root folder of dataset', default='data_path/NYUD')
# parser.add_argument('--fold', default='3', type=str,
#                     help='fold')
parser.add_argument('--output_dir', help='root folder of output', default='./output/sauge/ss/')
parser.add_argument('--model_path', default="./ckpt/bsds/sauge_vitb.pth", type=str, help='the path of check point')
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.device
def init_seeds(seed=0, cuda_deterministic=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def save_checkpoint(state, filename='checkpoint.pth'):
    torch.save(state, filename)


def main():
    sam = sam_model_registry[args.sam_type](checkpoint=args.sam_ckpt)
    for name, parameter in sam.named_parameters():
        parameter.requires_grad = False
    sam.cuda()
    num_gpus = torch.cuda.device_count()
    mask_generator = SamAutomaticMaskGeneratorForTest(sam)
    args.cuda = True
    test_dataset = BSDS_Loader(root=args.dataset,  split= "test")
    # test_dataset = Multicue_Loader(type='edges', fold=args.fold, root=args.dataset,  split= "test")
    # test_dataset = NYUD_Loader(root=args.dataset,  split= "test")
    test_loader = DataLoader(
        test_dataset, batch_size=1,
        num_workers=4, drop_last=True,shuffle=False)
    # with open('data_path/test_edges_all_{}.lst'.format(args.fold), 'r') as f:
    #     test_list = f.readlines()
    # with open('data_path/multicue/test_edges_all.lst', 'r') as f:
    #     test_list = f.readlines()
    # with open('data_path/NYUD/image-test.lst') as f:
    #     test_list = f.readlines()
    with open('{}/test.lst'.format(args.dataset)) as f:
        test_list = f.readlines()
    test_list = [split(i.rstrip())[1] for i in test_list]
    assert len(test_list) == len(test_loader), "%d vs %d" % (len(test_list), len(test_loader))

    # model
    model = Model.SAUGE(args, sam_generator=mask_generator, mode='eval').cuda()
    checkpoint = torch.load(args.model_path)['state_dict']

    model.load_state_dict(checkpoint)
    model.eval()
    
    total_params = sum(p.numel() for p in model.parameters())
    total_params += sum(p.numel() for p in model.buffers())
    print(f'{total_params:,} total parameters.')
    print(f'{total_params/(1024*1024):.2f}M total parameters.')
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad)
    print(f'{total_trainable_params:,} training parameters.')
    print(f'{total_trainable_params/(1024*1024):.2f}M training parameters.')
    
    alpha = 0
    # multicue_test(model, test_loader, alpha, test_list=test_list,
    #             save_dir = join(args.output_dir, 'test-version'))
    test(model, test_loader, alpha, test_list=test_list,
                save_dir = join(args.output_dir, 'predict-view'), bsds=True)



def test(model, test_loader, alpha, test_list, save_dir, bsds):
    # model.eval()
    alpha_list = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    if not isdir(save_dir):
        os.makedirs(save_dir)
    for idx, (image) in enumerate(tqdm(test_loader)):
        image = image.cuda()
        with torch.no_grad():
            multi_outputs, outputs, _, _= model(image)
        png=torch.squeeze(outputs.detach()).cpu().numpy()
        _, _, H, W = image.shape
        if not bsds:
            result = png
        else:
            result=np.zeros((H+1,W+1))
            result[1:,1:]=png
        filename = splitext(test_list[idx])[0]
        result_png = Image.fromarray((result * 255).astype(np.uint8))
        
        png_save_dir=os.path.join(save_dir,"png")
        mat_save_dir=os.path.join(save_dir,"mat")

        if not os.path.exists(png_save_dir):
            os.makedirs(png_save_dir)

        if not os.path.exists(mat_save_dir):
            os.makedirs(mat_save_dir)
        result_png.save(join(png_save_dir, "%s.png" % filename))
        io.savemat(join(mat_save_dir, "%s.mat" % filename),{'result':result},do_compression=True)
        
        # multi-granularity output
        for alpha in alpha_list:
            if alpha <= 0.5:
                percent_1 = alpha / 0.5
                percent_2 = 1 - percent_1
                any_output = percent_1 * multi_outputs[1] + percent_2 * multi_outputs[0]
                any_output = torch.clamp(any_output, 0, 1)
            else:
                percent_1 = (alpha - 0.5) / 0.5
                percent_2 = 1 - percent_1
                any_output = percent_1 * multi_outputs[2] + percent_2 * multi_outputs[1]
                any_output = torch.clamp(any_output, 0, 1)
            png=torch.squeeze(any_output.detach()).cpu().numpy()
            if not bsds:
                result = png
            else:
                result=np.zeros((H+1,W+1))
                result[1:,1:]=png
            filename = splitext(test_list[idx])[0]
            result_png = Image.fromarray((result * 255).astype(np.uint8))
            
            png_save_dir=os.path.join(save_dir + '/' + str(alpha),"png")
            mat_save_dir=os.path.join(save_dir + '/' + str(alpha),"mat")

            if not os.path.exists(png_save_dir):
                os.makedirs(png_save_dir)

            if not os.path.exists(mat_save_dir):
                os.makedirs(mat_save_dir)
            result_png.save(join(png_save_dir, "%s.png" % filename))
            io.savemat(join(mat_save_dir, "%s.mat" % filename),{'result':result},do_compression=True)


def multicue_test(model, test_loader, epoch, test_list, save_dir):
    alpha_list = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    for idx, image in enumerate(tqdm(test_loader)):
        line=test_list[idx].strip("\n").split("\t")
        image = image.cuda()
        with torch.no_grad():
            
            batch_size, _, h_img, w_img = image.size()
            h_crop = 512
            w_crop = 512
            h_stride = 300
            w_stride = 200
            
            h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
            w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
            preds = image.new_zeros((batch_size, 1, h_img, w_img))
            preds_corase = image.new_zeros((batch_size, 1, h_img, w_img))
            preds_middle = image.new_zeros((batch_size, 1, h_img, w_img))
            preds_fine = image.new_zeros((batch_size, 1, h_img, w_img))
            count_mat = image.new_zeros((batch_size, 1, h_img, w_img))
            for h_idx in range(h_grids):
                for w_idx in range(w_grids):
                    y1 = h_idx * h_stride
                    x1 = w_idx * w_stride
                    y2 = min(y1 + h_crop, h_img)
                    x2 = min(x1 + w_crop, w_img)
                    y1 = max(y2 - h_crop, 0)
                    x1 = max(x2 - w_crop, 0)
                    crop_img = image[:, :, y1:y2, x1:x2]
                    with torch.no_grad():
                        multi_outputs, outputs, _, _= model(crop_img)
                    preds += F.pad(outputs,
                                (int(x1), int(preds.shape[3] - x2), int(y1),
                                int(preds.shape[2] - y2)))
                    preds_corase += F.pad(multi_outputs[0],
                                (int(x1), int(preds_corase.shape[3] - x2), int(y1),
                                int(preds_corase.shape[2] - y2)))
                    preds_middle += F.pad(multi_outputs[1],
                                (int(x1), int(preds_middle.shape[3] - x2), int(y1),
                                int(preds_middle.shape[2] - y2)))
                    preds_fine += F.pad(multi_outputs[2],
                                (int(x1), int(preds_fine.shape[3] - x2), int(y1),
                                int(preds_fine.shape[2] - y2)))
                    count_mat[:, :, y1:y2, x1:x2] += 1
            assert (count_mat == 0).sum() == 0
            preds = preds / count_mat
            preds_corase = preds_corase / count_mat
            preds_middle = preds_middle / count_mat
            preds_fine = preds_fine / count_mat

            result=torch.squeeze(preds.detach()).cpu().numpy()
            result_c=torch.squeeze(preds_corase.detach()).cpu().numpy()
            result_m=torch.squeeze(preds_middle.detach()).cpu().numpy()
            result_f=torch.squeeze(preds_fine.detach()).cpu().numpy()

            filename = splitext(test_list[idx])[0]
            result_png = Image.fromarray((result * 255).astype(np.uint8))

            png_save_dir=os.path.join(save_dir,"png")
            mat_save_dir=os.path.join(save_dir,"mat")
            os.makedirs(png_save_dir,exist_ok=True)
            os.makedirs(mat_save_dir,exist_ok=True)
            result_png.save(join(png_save_dir, "%s.png" % filename))
            io.savemat(join(mat_save_dir, "%s.mat" % filename),{'result':result},do_compression=True)


            # multi-granularity output
            for alpha in alpha_list:
                if alpha <= 0.5:
                    percent_1 = alpha / 0.5
                    percent_2 = 1 - percent_1
                    any_output = percent_1 * result_m + percent_2 * result_c
                    any_output = np.clip(any_output, 0, 1)
                else:
                    percent_1 = (alpha - 0.5) / 0.5
                    percent_2 = 1 - percent_1
                    any_output = percent_1 * result_f + percent_2 * result_m
                    any_output = np.clip(any_output, 0, 1)

                filename = splitext(test_list[idx])[0]
                result_png = Image.fromarray((any_output * 255).astype(np.uint8))
                
                png_save_dir=os.path.join(save_dir, str(alpha),"png")
                mat_save_dir=os.path.join(save_dir, str(alpha),"mat")

                if not os.path.exists(png_save_dir):
                    os.makedirs(png_save_dir)

                if not os.path.exists(mat_save_dir):
                    os.makedirs(mat_save_dir)
                result_png.save(join(png_save_dir, "%s.png" % filename))
                io.savemat(join(mat_save_dir, "%s.mat" % filename),{'result':any_output},do_compression=True)



if __name__ == '__main__':
    random_seed = 6666
    init_seeds(random_seed)
    main()
   
