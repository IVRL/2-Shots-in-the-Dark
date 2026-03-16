import os
import time
import logging
import math
import argparse
import random
import sys
import numpy as np
import glob
from collections import OrderedDict
import pickle
import rawpy
from PIL import Image
import exifread
from torch.autograd import Variable
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.utils.data as data
from skimage.metrics import structural_similarity, peak_signal_noise_ratio
# import lpips
from utils.util import setup_logger, print_args, Logger
from models.modules import define_G
from utils import raw_util, metric_util
from models.trainer_denoising import Trainer


lrid_folder = '/scratch/students/2023-fall-sp-liying/dataset/LRID/'
indoor_x5_ratio_list = [1, 2, 4, 8, 16]
indoor_x5_scene_list = [4, 14, 25, 41, 44, 51, 52, 53, 58]
# outdoor_x3
outdoor_x3_ratio_list = [1, 2, 4]
outdoor_x3_scene_list = [9, 21, 22, 32, 44, 51]

ratio_map = {64:1, 128:2, 256:4, 512:8, 1024:16}


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_networks(network, resume, device, strict=True):
    load_path = resume
    if isinstance(network, nn.DataParallel) or isinstance(network, DistributedDataParallel):
        network = network.module
    load_net = torch.load(load_path, map_location=torch.device(device))
    load_net_clean = OrderedDict()  # remove unnecessary 'module.'
    for k, v in load_net.items():
        if k.startswith('module.'):
            load_net_clean[k[7:]] = v
        else:
            load_net_clean[k] = v
    if 'optimizer' or 'scheduler' in net_name:
        network.load_state_dict(load_net_clean)
    else:
        network.load_state_dict(load_net_clean, strict=strict)

def get_ISO_ExposureTime(filepath):
    raw_file = open(filepath, 'rb')
    exif_file = exifread.process_file(raw_file, details=False, strict=True)
    
    if 'EXIF ExposureTime' in exif_file:
        exposure_str = exif_file['EXIF ExposureTime'].printable
    else:
        exposure_str = exif_file['Image ExposureTime'].printable
    if '/' in exposure_str:
        fenmu = float(exposure_str.split('/')[0])
        fenzi = float(exposure_str.split('/')[-1])
        exposure = fenmu / fenzi
    else:
        exposure = float(exposure_str)

    if 'EXIF ISOSpeedRatings' in exif_file:
        ISO_str = exif_file['EXIF ISOSpeedRatings'].printable
    else:
        ISO_str = exif_file['Image ISOSpeedRatings'].printable
    if '/' in ISO_str:
        fenmu = float(ISO_str.split('/')[0])
        fenzi = float(ISO_str.split('/')[-1])
        ISO = fenmu / fenzi
    else:
        ISO = float(ISO_str)

    exposure = exposure * 1000
    
    return int(ISO), exposure

    
def load_darkshading(iso, exp, naive=True, hot=False):
    def bayer2rggb(bayer):
        H, W = bayer.shape
        return bayer.reshape(H//2, 2, W//2, 2).transpose(0, 2, 1, 3).reshape(H//2, W//2, 4)
    def rggb2bayer(rggb):
        H, W, _ = rggb.shape
        return rggb.reshape(H, W, 2, 2).transpose(0, 2, 1, 3).reshape(H*2, W*2)
    def blc_rggb(raw, bias):
        return rggb2bayer(bayer2rggb(raw) + bias.reshape(1,1,4))    
    def get_bias(iso, exp, blc_mean):
        bias = blc_mean[iso][:,0] * exp + blc_mean[iso][:,1] # RGGB: (4,)
        return bias
        
    if naive:
        ds_path = os.path.join(lrid_folder, 'resources', f'darkshading-iso-{iso}.npy')
        darkshading = np.load(ds_path)
        darkshading_hot = np.load(ds_path[:-4]+'-hot.npy')
        
    else:
        ds_tk = np.load(os.path.join(lrid_folder, 'resources', f'darkshading_tk.npy'))
        ds_tk_hot = np.load(os.path.join(lrid_folder, 'resources', f'darkshading_tk_hot.npy'))
        ds_tb = np.load(os.path.join(lrid_folder, 'resources', f'darkshading_tb.npy'))
        ds_tb_hot= np.load(os.path.join(lrid_folder, 'resources', f'darkshading_tb_hot.npy'))
        # rggb, k*exp+b
        with open(os.path.join(lrid_folder, 'resources', f'BLE_t.pkl'),'rb') as f:
            blc_mean = pickle.load(f)
        with open(os.path.join(lrid_folder, 'resources', f'BLE_t_hot.pkl'),'rb') as f:
            blc_mean_hot = pickle.load(f)

        if hot:
            darkshading_hot = ds_tk_hot * 30 + ds_tb_hot
            bias_hot = get_bias(iso, 30, blc_mean_hot)
            darkshading_hot = blc_rggb(darkshading_hot, bias_hot)
        else:
            darkshading = ds_tk * 30 + ds_tb
            bias = get_bias(iso, 30, blc_mean)
            darkshading = blc_rggb(darkshading, bias)

    if naive:
        ds = darkshading_hot if hot else darkshading
    else:
        ds = darkshading_hot if hot else darkshading
        if hot:
            bias_delta = get_bias(iso, exp, blc_mean_hot) - get_bias(iso, 30, blc_mean_hot)
        else:
            bias_delta = get_bias(iso, exp, blc_mean) - get_bias(iso, 30, blc_mean)
        ds = ds + bias_delta.mean()
        
        
    return ds



def hot_check(condition_folder, scene_id):
    hot_ids = []
    if condition_folder == 'indoor_x5':
        hot_ids = [6,15,33,35,39,46,37,59]
    elif condition_folder == 'indoor_x3':
        hot_ids = [1,2,4,5,6,10,12,13,14,15,16,17,18,19]
    elif condition_folder == 'outdoor_x3':
        hot_ids = [0,1,2,3,4,5,7,10,11,12,13,14,15,16,17,18,19,22,26,30,51,52,54,55,56]
    elif condition_folder == 'outdoor_x5':
        hot_ids = [0,1,2,3,4,5,6]

    hot = True if scene_id in hot_ids else False
    
    return hot

def get_darkshading_from_singleimage(hot):
    if hot:
        folder = 'notebook/LRID_Hot_DarkShadings_BlurSigma50'
    else:
        folder = 'notebook/LRID_DarkShadings_BlurSigma50'
    path = sorted(glob.glob(os.path.join(folder, '*.npy')))[0]
    darkshading = np.load(path)
    return darkshading


    
def load_all_image_info(condition='indoor', ratio=1, use_realdarkshading=True):
    in_paths = []
    gt_paths = []
    darkshading_list = []
    
    if condition == 'indoor':
        condition_folder = 'indoor_x5'
        scene_list = indoor_x5_scene_list
    elif condition == 'outdoor':
        condition_folder = 'outdoor_x3'
        scene_list = outdoor_x3_scene_list
    for scene in scene_list:
        in_folder = os.path.join(lrid_folder, condition_folder, '6400', str(ratio), f"{scene:03d}")
        in_path = sorted(glob.glob(os.path.join(in_folder, '*.dng')))[0]
        gt_path = os.path.join(lrid_folder, condition_folder, 'npy/GT_align_ours', f"{scene:03d}.npy")
        hot = hot_check(condition_folder, scene)
        iso, exp = get_ISO_ExposureTime(in_path)
        if use_realdarkshading:
            darkshading = load_darkshading(iso=iso, exp=exp, naive=False, hot=hot)
        else:
            darkshading = get_darkshading_from_singleimage(hot)
        
        in_paths.append(in_path)
        gt_paths.append(gt_path)
        darkshading_list.append(darkshading)
        
    return in_paths, gt_paths, darkshading_list


def pack_raw(im, darkshading, bl=64., wl=1023., ds_correction=True):
    # to align with training
    if ds_correction:
        im = im - darkshading

    im = np.expand_dims(im, axis=2)
    img_shape = im.shape
    H = img_shape[0]
    W = img_shape[1]

    # pack Bayer image to 4 channels
    out = np.concatenate((im[0:H:2, 0:W:2, :], # r
                          im[0:H:2, 1:W:2, :], # g
                          im[1:H:2, 1:W:2, :], # b
                          im[1:H:2, 0:W:2, :]), # g
                         axis=2).astype(np.float32)
    
    out = out - bl # subtract the black level
    out = out / (wl - bl) 

    return out.astype(np.float32) 


def raw2bayer(raw, wp=1023, bl=64, norm=True, clip=False, bias=np.array([0,0,0,0])):
    raw = raw.astype(np.float32)
    H, W = raw.shape
    out = np.stack((raw[0:H:2, 0:W:2], #RGBG
                    raw[0:H:2, 1:W:2],
                    raw[1:H:2, 1:W:2],
                    raw[1:H:2, 0:W:2]), axis=0).astype(np.float32) 
    if norm:
        bl = bias + bl
        bl = bl.reshape(4, 1, 1) 
        out = (out - bl) / (wp - bl)
    if clip: out = np.clip(out, 0, 1)
        
    return out.astype(np.float32) 
    

def load_image(in_path, gt_path, darkshading, ratio, use_realdarkshading=False):
    # read raw images
    raw = rawpy.imread(in_path)
    raw = raw.raw_image_visible.astype(np.float32)
    gt_raw = np.load(gt_path).astype(np.float32)

    if use_realdarkshading:
        raw = raw - darkshading
        
    input_norm = raw2bayer(raw, norm=True, clip=False)
    gt_norm = raw2bayer(gt_raw, norm=True, clip=True)

    if not use_realdarkshading:
        input_norm = input_norm - darkshading

    input_norm = input_norm * ratio
    
    sample = {
              'noisy_img': input_norm,
              'clean_img': gt_norm,
             }

    for key in sample.keys():
        sample[key] = torch.from_numpy(sample[key].astype(np.float32)).float()
        sample[key] = sample[key] #.permute(2, 0, 1)


    return sample, raw, gt_raw



def tensor2im(image_tensor, visualize=False, video=False):    
    image_tensor = image_tensor.detach()

    if visualize:                
        image_tensor = image_tensor[:, 0:3, ...]

    if not video: 
        image_numpy = image_tensor[0].cpu().float().numpy()
        image_numpy = (np.transpose(image_numpy, (1, 2, 0))) #* 255.0
    else:
        image_numpy = image_tensor.cpu().float().numpy()
        image_numpy = (np.transpose(image_numpy, (0, 2, 3, 1))) #* 255.0

    image_numpy = np.clip(image_numpy, 0, 1)

    return image_numpy


def crop_center(img,cropx,cropy):
    _, _, y, x = img.shape
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)
    return img[:, :, starty:starty+cropy,startx:startx+cropx]



def quality_assess(X, Y, data_range=255):
    # Y: correct; X: estimate
    if X.ndim == 3:  # image
        psnr = peak_signal_noise_ratio(Y, X, data_range=data_range)
        ssim = structural_similarity(Y, X, data_range=data_range, channel_axis=2)
        return {'PSNR':psnr, 'SSIM': ssim}

    else:
        raise NotImplementedError



class IlluminanceCorrect(nn.Module):
    def __init__(self):
        super(IlluminanceCorrect, self).__init__()
    
    # Illuminance Correction
    def forward(self, predict, source):
        if predict.shape[0] != 1:
            output = torch.zeros_like(predict)
            if source.shape[0] != 1:
                for i in range(predict.shape[0]):
                    output[i:i+1, ...] = self.correct(predict[i:i+1, ...], source[i:i+1, ...])               
            else:                                     
                for i in range(predict.shape[0]):
                    output[i:i+1, ...] = self.correct(predict[i:i+1, ...], source)                    
        else:
            output = self.correct(predict, source)
        return output

    def correct(self, predict, source):
        N, C, H, W = predict.shape        
        predict = torch.clamp(predict, 0, 1)
        assert N == 1
        output = torch.zeros_like(predict, device=predict.device)
        pred_c = predict[source != 1]
        source_c = source[source != 1]
        
        num = torch.dot(pred_c, source_c)
        den = torch.dot(pred_c, pred_c)        
        output = num / den * predict
        # print(num / den)

        return output

    

def postprocess_bayer(rawpath, img4c):
    img4c = img4c.detach()
    img4c = img4c[0].cpu().float().numpy()
    img4c = np.clip(img4c, 0, 1)

    #unpack 4 channels to Bayer image
    raw = rawpy.imread(rawpath)
    raw_pattern = raw.raw_pattern
    R = np.where(raw_pattern==0)
    G1 = np.where(raw_pattern==1)
    G2 = np.where(raw_pattern==3)
    B = np.where(raw_pattern==2)
    
    black_level = np.array(raw.black_level_per_channel)[:,None,None]

    white_point = 1023

    img4c = img4c * (white_point - black_level) + black_level
    
    img_shape = raw.raw_image_visible.shape
    H = img_shape[0]
    W = img_shape[1]

    raw.raw_image_visible[R[0][0]:H:2, R[1][0]:W:2] = img4c[0, :,:]
    raw.raw_image_visible[G1[0][0]:H:2,G1[1][0]:W:2] = img4c[1, :,:]
    raw.raw_image_visible[B[0][0]:H:2,B[1][0]:W:2] = img4c[2, :,:]
    raw.raw_image_visible[G2[0][0]:H:2,G2[1][0]:W:2] = img4c[3, :,:]
    
    out = raw.postprocess(use_camera_wb=True, half_size=True, no_auto_bright=True, output_bps=8, bright=1, user_black=None, user_sat=None)
    
    return out



def process_image_pair(in_path, gt_path, darkshading, ratio, net, device, corrector, args):
    # Load and prepare the image pair
    save_folder = args.save_folder
    sample, raw, raw_gt = load_image(in_path, gt_path, darkshading, ratio, args.use_realdarkshading)
    
    # Move data to device and add batch dimension
    for key in sample:
        sample[key] = Variable(sample[key].to(device), requires_grad=False)
        sample[key] = sample[key].unsqueeze(0)
    
    noisy_img = sample['noisy_img']
    clean_img = sample['clean_img']

    
    _, _, h, w = noisy_img.shape
    if h % 16 != 0 or w % 16 != 0:
        # align with PMN paper
        pad_top, pad_bottom, pad_left, pad_right = 4, 4, 4, 4
        p2d = (pad_left, pad_right, pad_top, pad_bottom)
        pad_noisy_img = F.pad(noisy_img, p2d, mode='reflect')
    else:
        pad_noisy_img = noisy_img

    # Network inference
    with torch.no_grad():
        output = net(pad_noisy_img)

    output = output[:, :, 4:-4, 4:-4]
    
    output = output.clamp(0.0, 1.0)
    clean_img = clean_img.clamp(0.0, 1.0)
    
    # Illumination correction
    if args.correct_illum:
        output = corrector(output, clean_img)
    
    # Calculate metrics
    output_np = tensor2im(output)
    target = tensor2im(clean_img)
    res = quality_assess(output_np, target, data_range=1)
    
    # Save processed image if requested
    if args.visualize_img:
        name_split = in_path.split('/')
        image_name = name_split[-5] + '_Ratio' +  name_split[-3] + '_Scene' + name_split[-2]
        
        output_processed = postprocess_bayer(in_path, output)
        Image.fromarray(output_processed.astype(np.uint8)).save(os.path.join(save_folder, f"{image_name}_output.png"))
        # clean_img = postprocess_bayer(in_path, clean_img)
        # Image.fromarray(clean_img.astype(np.uint8)).save(os.path.join(args.save_folder, f"{image_name}_clean.png"))
        # noisy_img = postprocess_bayer(in_path, noisy_img)
        # Image.fromarray(noisy_img.astype(np.uint8)).save(os.path.join(args.save_folder, f"{image_name}_noisy.png"))
    
    return res


    
def main():
    parser = argparse.ArgumentParser(description='LRDI Denoising Testing')
    parser.add_argument('--random_seed', default=0, type=int)
    parser.add_argument('--phase', default='test', type=str)

    ## device setting
    parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
    parser.add_argument('--launcher', choices=['none', 'pytorch'], default='none',
                        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    
    ## estimation
    parser.add_argument('--noise_param_estm', action='store_true')
    parser.add_argument('--visualize_img', action='store_true')
    parser.add_argument('--correct_illum', action='store_true')
    parser.add_argument('--correct_darkshading', action='store_true')
    parser.add_argument('--use_realdarkshading', action='store_true')
    
    ## network setting
    parser.add_argument('--net_name', default='LSID', type=str, help='UNetSeeInDark | LSID')

    ## dataloader setting
    parser.add_argument('--condition', type=str, default='indoor')
    parser.add_argument('--ratio', type=int, default=1)
    parser.add_argument('--test_dataset', default='SID', type=str, help='SID | ELD')

    parser.add_argument('--resume', default='', type=str)
    
    parser.add_argument('--save_folder', default='./logs/denoising_new/results_LRID', type=str)
    
    
    ## Setup training environment
    args = parser.parse_args()
    set_random_seed(args.random_seed)
    

    ## Setup training device
    str_ids = args.gpu_ids.split(',')
    args.gpu_ids = []
    for str_id in str_ids:
        id = int(str_id)
        if id >= 0:
            args.gpu_ids.append(id)
    if len(args.gpu_ids) > 0:
        torch.cuda.set_device(args.gpu_ids[0])
    device = torch.device('cuda' if len(args.gpu_ids) != 0 else 'cpu')
    args.device = device

    ## Distributed settings
    if args.launcher == 'none': 
        args.dist = False
        args.rank = -1
    else:
        args.dist = True
        init_dist()
        args.world_size = torch.distributed.get_world_size()
        args.rank = torch.distributed.get_rank()

    ## Setup image saving path
    if args.visualize_img:
        args.save_folder = os.path.join(args.save_folder, args.resume.split('/')[-3])
        if not os.path.exists(args.save_folder):
            os.makedirs(args.save_folder)

        log_path = os.path.join(args.save_folder, f"test_results_{args.condition}_Ratio{args.ratio}.txt")
        log_file = open(log_path, "w")
        sys.stdout = Logger(log_file)

    print_args(args)
    cudnn.benchmark = True
    

    ## Init network
    net = define_G(args)
    if args.resume:
        load_networks(net, args.resume, device)
    net.eval()


    test_ratio = args.ratio
    in_paths, gt_paths, darkshading_list = load_all_image_info(condition=args.condition, 
                                                               ratio=test_ratio, 
                                                               use_realdarkshading=args.use_realdarkshading)

    
    corrector = IlluminanceCorrect()

    test_ratio = args.ratio
    psnr, ssim = [], []

    ## Iterate over test samples
    for img_idx in range(len(in_paths)):
        in_path, gt_path, darkshading = in_paths[img_idx], gt_paths[img_idx], darkshading_list[img_idx]
            
        res = process_image_pair(
            in_path, gt_path, darkshading, test_ratio, net, device, 
            corrector, args
        )

        # Record metrics and print results
        psnr.append(res['PSNR'])
        ssim.append(res['SSIM'])
        print_name = in_path.split('/')
        print_name = os.path.join(print_name[-5], print_name[-4], print_name[-3], print_name[-2])
        print(f"Ratio: {test_ratio}, Path: {print_name} -- PSNR / SSIM: {res['PSNR']:.4f} / {res['SSIM']:.4f}")

    print("===> Averaged PSNR / SSIM: {} / {}".format(np.array(psnr).mean(), np.array(ssim).mean()))




if __name__ == '__main__':
    main()
