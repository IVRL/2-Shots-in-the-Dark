import os
# from imageio import imread
from PIL import Image, ImageOps
import numpy as np
import glob
import random
import math
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms
import rawpy
import scipy
from scipy.stats import truncnorm
import cv2
import itertools
import sys
sys.path.append('..')
from utils import util
from utils import raw_util


train_path = "/scratch/students/2023-fall-sp-liying/dataset/SID/Sony_train_list.txt"
data_folder = "/scratch/students/2023-fall-sp-liying/dataset/SID"
synthetic_folder = './NoiseDiff_GeneratedNoiseData'


class DenoisingDataset_RealData_RemoveDarkShading(Dataset):
    def __init__(self, args):
        self.args = args
        iso_value = args.iso_value
        ratio_value = args.ratio_value
        
        pair_list = []

        # get paths of dark frame files
        self.darkframe_paths = self.load_darkframe_paths()
        iso_available = list(self.darkframe_paths.keys())
        self.iso_available = iso_available
        
        
        # real data
        train_path = "/scratch/students/2023-fall-sp-liying/dataset/SID/Sony_train_list.txt"
        data_folder = "/scratch/students/2023-fall-sp-liying/dataset/SID"
        
        bad_raw = []
        with open(train_path, 'r') as file:
            for line in file:
                if line:
                    in_path, gt_path, iso, fvalue = line.split(' ')
                    iso = int(iso.replace('ISO', ''))
                    in_fn = os.path.basename(in_path)
                    gt_fn = os.path.basename(gt_path)
                    test_id = int(in_fn[0:5])
                    in_exposure = float(in_fn[9:-5])
                    gt_exposure = float(gt_fn[9:-5])
                    ratio = min(gt_exposure / in_exposure, 300)
                    
                    in_path = os.path.join(data_folder, in_path)
                    gt_path = os.path.join(data_folder, gt_path)
                    
                    # if math.isclose(in_exposure, 0.033, abs_tol=1e-3) and iso in iso_available:
                    # if iso == iso_value:
                    try:
                        gt_raw = rawpy.imread(gt_path)
                        clean_img = raw_util.pack_raw(gt_raw, rescale=False)
                        clean_img = clean_img.transpose(2,0,1)
    
                        raw = rawpy.imread(in_path)
                        noisy_img = raw_util.pack_raw(raw, rescale=False) 
                        noisy_img = noisy_img.transpose(2,0,1)
                        
                        pair_list.append([clean_img, noisy_img, iso, ratio])
                        
                    except:
                        bad_raw.append(in_path)
                        continue


        print('----- bad_raw -----:', bad_raw)                        
        self.data_len = len(pair_list)
        if self.data_len == 0:
            print('No data found!')
            sys.exit()

        self.pair_list = pair_list
        print('image number: ', len(self.pair_list))
        
        # read noise profile
        with open('/scratch/students/2023-fall-sp-liying/code/FlowMatching/notebook/noise_profile_all.pkl', 'rb') as file:
            self.noise_profile = pickle.load(file)

        # load dark shadings
        self.darkshading_dict = self.load_darkshadings()


    def __len__(self):
        return self.data_len

        
    def load_darkshadings(self):
        darkshading_dict = {}
        for iso in self.iso_available:
            darkshading, _ = raw_util.get_darkshading_from_average(iso)  # darkshading was normalized to [0,1]
            darkshading = darkshading * (16383 - 512)
            darkshading = darkshading.transpose(2,0,1)
            darkshading_dict[iso] = darkshading
        return darkshading_dict

        
    def load_darkframe_paths(self):
        folders = glob.glob(os.path.join("/scratch/students/2023-fall-sp-liying/dataset/Sony_Bias_Frame/ISO*"))
        darkframe_dict = {}
        for folder in folders:
            iso = int(folder.split('/')[-1].replace('ISO', ''))
            mat_files = list(sorted(glob.glob(os.path.join(folder, "*.mat"))))
            darkframe_dict[iso] = mat_files

        return darkframe_dict


    def aug(self, img_list, h, w):
        _, ih, iw = img_list[1].shape
        
        x = np.random.randint(0, iw - w + 1)
        y = np.random.randint(0, ih - h + 1)
        x = x // 2 * 2
        y = y // 2 * 2
        for i in range(len(img_list)):
            img_list[i] = img_list[i][:, y:y+h, x:x+w]
            
        return img_list
        
        

    def __getitem__(self, idx):
        clean_img, noisy_img, iso, ratio = self.pair_list[idx]
        darkshading = self.darkshading_dict[iso]

        clean_img, noisy_img, darkshading = self.aug([clean_img, noisy_img, darkshading],
                             self.args.crop_size, self.args.crop_size)
        noisy_img = noisy_img - darkshading

        noisy_img = noisy_img * ratio
        noisy_img = noisy_img.clip(0, 16383 - 512)
        
        clean_img = clean_img / (16383 - 512)
        noisy_img = noisy_img / (16383 - 512)

        sample = {
                  'clean_img': clean_img,
                  'noisy_img': noisy_img,
                 }

        for key in sample.keys():
            if key not in ['iso', 'ratio', 'iso_ratio_idx', 'noisy_name', 'clean_name']:
                sample[key] = sample[key].astype(np.float32)
                sample[key] = torch.from_numpy(sample[key]).float()

        return sample




class DenoisingDataset_PossionWithRealDarkframes_RemoveDarkShading(Dataset):
    def __init__(self, args):
        self.args = args
        iso_value = args.iso_value
        ratio_value = args.ratio_value
        
        pair_list = []

        # get paths of dark frame files
        self.darkframe_paths = self.load_darkframe_paths()
        iso_available = list(self.darkframe_paths.keys())
        self.iso_available = iso_available
        
        
        # real data
        train_path = "/scratch/students/2023-fall-sp-liying/dataset/SID/Sony_train_list.txt"
        data_folder = "/scratch/students/2023-fall-sp-liying/dataset/SID"

        clean_img_dict = {}
        pair_list_temp = []
        with open(train_path, 'r') as file:
            for line in file:
                if line:
                    in_path, gt_path, iso, fvalue = line.split(' ')
                    iso = int(iso.replace('ISO', ''))
                    in_fn = os.path.basename(in_path)
                    gt_fn = os.path.basename(gt_path)
                    test_id = int(in_fn[0:5])
                    in_exposure = float(in_fn[9:-5])
                    gt_exposure = float(gt_fn[9:-5])
                    ratio = min(gt_exposure / in_exposure, 300)
                    
                    in_path = os.path.join(data_folder, in_path)
                    gt_path = os.path.join(data_folder, gt_path)
                    
                    # # if math.isclose(in_exposure, 0.033, abs_tol=1e-3) and iso in iso_available:
                    # if iso == iso_value:
                    #     gt_raw = rawpy.imread(gt_path)
                    #     clean_img = raw_util.pack_raw(gt_raw, rescale=False)
                    #     clean_img = clean_img.transpose(2,0,1)
                    #     pair_list.append([clean_img, iso, ratio])

                    if not gt_path in clean_img_dict.keys():
                        gt_raw = rawpy.imread(gt_path)
                        clean_img = raw_util.pack_raw(gt_raw, rescale=False)
                        clean_img = clean_img.transpose(2,0,1)
                        clean_img_dict[gt_path] = clean_img
                        
                    pair_list_temp.append([gt_path, iso, ratio])
                    
        for pair in pair_list_temp:
            gt_path, iso, ratio = pair
            clean_img = clean_img_dict[gt_path]
            pair_list.append([clean_img, iso, ratio])


                        
        self.data_len = len(pair_list)
        if self.data_len == 0:
            print('No data found!')
            sys.exit()

        self.pair_list = pair_list
        print('image number: ', len(self.pair_list))
        
        # read noise profile
        with open('/scratch/students/2023-fall-sp-liying/code/FlowMatching/notebook/noise_profile_all.pkl', 'rb') as file:
            self.noise_profile = pickle.load(file)

        # load darkshading dict
        self.darkshading_dict = self.load_darkshadings()

    def __len__(self):
        return self.data_len #+ 15*3


    def load_darkshadings(self):
        darkshading_dict = {}
        for iso in self.iso_available:
            statc_folder = '/scratch/students/2023-fall-sp-liying/code/FlowMatching/notebook/DarkShading_%d' % self.args.darkframe_num
            if not os.path.exists(statc_folder):
                print('darkframe_num: %d not available!!!' % self.args.darkframe_num)
                
            darkshading = np.load(os.path.join(statc_folder, 'ISO%d_mean.npy' % iso))  # shape: (1, 4, H, W)
            darkshading = darkshading[0]
    
            darkshading = darkshading * (16383 - 512)
            
            darkshading_dict[iso] = darkshading
        return darkshading_dict

        
    def load_darkframe_paths(self):
        folders = glob.glob(os.path.join("/scratch/students/2023-fall-sp-liying/dataset/Sony_Bias_Frame/ISO*"))
        darkframe_dict = {}
        for folder in folders:
            iso = int(folder.split('/')[-1].replace('ISO', ''))
            mat_files = list(sorted(glob.glob(os.path.join(folder, "*.mat"))))
            ## Only use [darkframe_num] samples per ISO!!!
            darkframe_dict[iso] = mat_files[:self.args.darkframe_num]

        return darkframe_dict
        

    def select_random_darkframe(self, iso):
        darkframe_path_list = self.darkframe_paths[iso]
        darkframe_path = np.random.choice(darkframe_path_list)
        darkframe_info = scipy.io.loadmat(darkframe_path)
        darkframe = darkframe_info["Inoisy_crop"].astype(np.float32) - 512
        darkframe = raw_util.pack_np_raw(darkframe).transpose(2,0,1)  # (C, H, W)
        if darkframe_info['expo'] != 1/30:
            print('path: %s  exp: %f != 0.033', (darkframe_path, darkframe_info['expo']))
            sys.exit()

        return darkframe


    def aug(self, img_list, h, w):
        _, ih, iw = img_list[0].shape
        
        x = np.random.randint(0, iw - w + 1)
        y = np.random.randint(0, ih - h + 1)
        x = x // 2 * 2
        y = y // 2 * 2
        for i in range(len(img_list)):
            img_list[i] = img_list[i][:, y:y+h, x:x+w]
            
        return img_list
        

    def generate_truncated_normal(self, mean, variance, lower_bound, upper_bound, sample_size):
        std_dev = np.sqrt(variance)
        a = (lower_bound - mean) / std_dev
        b = (upper_bound - mean) / std_dev 
        truncated_samples = truncnorm.rvs(a, b, loc=mean, scale=std_dev, size=sample_size)

        return truncated_samples


    def apply_noise(self, clean, darkframe, darkshading, iso, ratio):
        K, VAR = self.noise_profile[iso]  # should be used on data after subtracting the black level, but without normalization

        latent = clean / float(ratio)
        
        C, H, W = latent.shape
        latent = latent.reshape(C*H*W)
        k = self.generate_truncated_normal(K, 1, lower_bound=0.7*K, upper_bound=1.3*K, sample_size=1)
        var = self.generate_truncated_normal(VAR, 1, lower_bound=0.7*VAR, upper_bound=1.3*VAR, sample_size=1)
        poisson = k * np.random.poisson(latent / k, size=C*H*W).reshape((C,H, W))
        # gaussian = np.random.normal(0, np.sqrt(var), C*H*W).reshape((C,H, W))
        # noisy = (poisson + gaussian) * ratio

        noisy = (poisson + darkframe) - darkshading
        noisy = noisy * ratio
        noisy = noisy.clip(0, 16383 - 512)
        
        return noisy
        

    def __getitem__(self, idx):
        clean_img, iso, ratio = self.pair_list[idx]
        darkframe = self.select_random_darkframe(iso)
        darkshading = self.darkshading_dict[iso]

        if self.args.randomcrop_darkshading:
            clean_img = self.aug([clean_img], self.args.crop_size, self.args.crop_size)[0]
            darkframe, darkshading = self.aug([darkframe, darkshading],
                                 self.args.crop_size, self.args.crop_size)
        else:
            clean_img, darkframe, darkshading = self.aug([clean_img, darkframe, darkshading],
                                 self.args.crop_size, self.args.crop_size)
        
        noisy_img = self.apply_noise(clean_img, darkframe, darkshading, iso, ratio)
        clean_img = clean_img / (16383 - 512)
        noisy_img = noisy_img / (16383 - 512)

        sample = {
                  'clean_img': clean_img,
                  'noisy_img': noisy_img,
                 }

        for key in sample.keys():
            if key not in ['iso', 'ratio', 'iso_ratio_idx', 'noisy_name', 'clean_name']:
                sample[key] = sample[key].astype(np.float32)
                sample[key] = torch.from_numpy(sample[key]).float()

        return sample





class DenoisingDataset_PossionWithSampledDarkframes_RemoveDarkShading(Dataset):
    def __init__(self, args):
        self.args = args
        iso_value = args.iso_value
        ratio_value = args.ratio_value
        
        pair_list = []

        # get paths of dark frame files
        self.darkframe_paths = self.load_darkframe_paths()
        iso_available = list(self.darkframe_paths.keys())
        self.iso_available = iso_available
        
        
        # real data
        train_path = "/scratch/students/2023-fall-sp-liying/dataset/SID/Sony_train_list.txt"
        data_folder = "/scratch/students/2023-fall-sp-liying/dataset/SID"
        
        gt_path_list = []
        clean_img_dict = {}
        pair_list_temp = []
        with open(train_path, 'r') as file:
            for line in file:
                if line:
                    in_path, gt_path, iso, fvalue = line.split(' ')
                    iso = int(iso.replace('ISO', ''))
                    in_fn = os.path.basename(in_path)
                    gt_fn = os.path.basename(gt_path)
                    test_id = int(in_fn[0:5])
                    in_exposure = float(in_fn[9:-5])
                    gt_exposure = float(gt_fn[9:-5])
                    ratio = min(gt_exposure / in_exposure, 300)
                    
                    in_path = os.path.join(data_folder, in_path)
                    gt_path = os.path.join(data_folder, gt_path)

                    # if not gt_path in gt_path_list:
                    #     gt_path_list.append(gt_path)
                    #     # exclude these two settings whose clean images contain noises
                    #     if iso != 12800 and iso != 25600:
                    #         gt_raw = rawpy.imread(gt_path)
                    #         clean_img = raw_util.pack_raw(gt_raw, rescale=False)
                    #         clean_img = clean_img.transpose(2,0,1)
                    #         # pair_list.append([clean_img, iso, ratio])
                    #         pair_list.append([clean_img, iso_value, np.random.choice([100., 250., 300.])])
                    #         if len(pair_list) >= 14*10:
                    #             break


                    if not gt_path in clean_img_dict.keys():
                        gt_raw = rawpy.imread(gt_path)
                        clean_img = raw_util.pack_raw(gt_raw, rescale=False)
                        clean_img = clean_img.transpose(2,0,1)
                        clean_img_dict[gt_path] = clean_img
                        
                    pair_list_temp.append([gt_path, iso, ratio])
                    
        for pair in pair_list_temp:
            gt_path, iso, ratio = pair
            clean_img = clean_img_dict[gt_path]
            pair_list.append([clean_img, iso, ratio])

                        
        self.data_len = len(pair_list)
        if self.data_len == 0:
            print('No data found!')
            sys.exit()

        self.pair_list = pair_list
        print('image number: ', len(self.pair_list))
        
        # read noise profile
        with open('/scratch/students/2023-fall-sp-liying/code/FlowMatching/notebook/noise_profile_all.pkl', 'rb') as file:
        # with open('/scratch/students/2023-fall-sp-liying/code/FlowMatching/notebook/noise_profile_from_single_noisy.pkl', 'rb') as file:
            self.noise_profile = pickle.load(file)

        # load darkshading dict
        self.darkshading_dict = self.load_darkshadings()


    def __len__(self):
        return self.data_len 


    def load_darkshadings(self):
        darkshading_dict = {}
        # files = sorted(glob.glob(os.path.join("/scratch/students/2023-fall-sp-liying/code/FlowMatching/notebook/SampledDarkShadings_ADSN_woGlobalMean_Averaged/*.npy")))
        # files = sorted(glob.glob(os.path.join("/scratch/students/2023-fall-sp-liying/code/FlowMatching/notebook/SampledDarkShadings_ADSN_Use400RealDF_IterHist/*.npy")))
        files = sorted(glob.glob(os.path.join(self.args.darkshading_folder, "/*.npy")))
        
        
        for file_path in files:
            darkshading = np.load(file_path)  # darkshading was normalized to [0,1]
            iso = int(os.path.basename(file_path).split('.')[0].replace('ISO', ''))
            
            darkshading_dict[iso] = darkshading
            
        return darkshading_dict

    def load_stds(self):
        std_dict = {}
        darkshading_dict = {}
        for iso in self.iso_available:
            darkshading, std = raw_util.get_darkshading_from_average(iso)
            std_dict[iso] = std.transpose(2,0,1)
            darkshading_dict[iso] = darkshading.transpose(2,0,1)
            
        return darkshading_dict, std_dict
        
    def load_darkframe_paths(self):
        # files = glob.glob(os.path.join("/scratch/students/2023-fall-sp-liying/code/FlowMatching/logs/flowmatching/results/test_0911_flowmatching_50files_allISO_samplefromprior_epoch30/npy/generated/*.npy"))
        # files = sorted(glob.glob(os.path.join("/scratch/students/2023-fall-sp-liying/code/FlowMatching/notebook/SampledDarkFrames_ADSN_Use400RealDF_IterHist/*.npy")))
        files = sorted(glob.glob(os.path.join(self.args.darkframe_folder, "*.npy")))
        darkframe_dict = {}
        for file_path in files:
            iso = int(os.path.basename(file_path).split('_')[0].replace('ISO', ''))
            if iso in darkframe_dict.keys():
                
                ## Only use 50 samples per ISO!!!
                if len(darkframe_dict[iso]) >= self.args.darkframe_num:
                    continue
                    
                darkframe_dict[iso].append(file_path)
            else:
                darkframe_dict[iso] = []

        return darkframe_dict
        

    def select_random_darkframe(self, iso):
        darkframe_path_list = self.darkframe_paths[iso]
        darkframe_path = np.random.choice(darkframe_path_list)
        darkframe = np.load(darkframe_path)  # (4, H, W), in the range of [0,1]

        return darkframe


    def aug(self, img_list, h, w):
        _, ih, iw = img_list[0].shape
        
        x = np.random.randint(0, iw - w + 1)
        y = np.random.randint(0, ih - h + 1)
        x = x // 2 * 2
        y = y // 2 * 2
        for i in range(len(img_list)):
            img_list[i] = img_list[i][:, y:y+h, x:x+w]
            
        return img_list
        

    def generate_truncated_normal(self, mean, variance, lower_bound, upper_bound, sample_size):
        std_dev = np.sqrt(variance)
        a = (lower_bound - mean) / std_dev
        b = (upper_bound - mean) / std_dev 
        truncated_samples = truncnorm.rvs(a, b, loc=mean, scale=std_dev, size=sample_size)

        return truncated_samples


    def apply_noise(self, clean, darkframe, iso, ratio):
        K, VAR = self.noise_profile[iso]  # should be used on data after subtracting the black level, but without normalization

        # same as 
        latent = clean / float(ratio)
        
        C, H, W = latent.shape
        latent = latent.reshape(C*H*W)
        k = self.generate_truncated_normal(K, 1, lower_bound=0.7*K, upper_bound=1.3*K, sample_size=1)
        # var = self.generate_truncated_normal(VAR, 1, lower_bound=0.7*VAR, upper_bound=1.3*VAR, sample_size=1)
        poisson = k * np.random.poisson(latent / k, size=C*H*W).reshape((C,H, W))
        # gaussian = np.random.normal(0, np.sqrt(var), C*H*W).reshape((C,H, W))
        # noisy = (poisson + gaussian) * ratio

        noisy = (poisson + darkframe)
        noisy = noisy * ratio
        noisy = noisy.clip(0, 16383 - 512)
        
        return noisy
        

    def __getitem__(self, idx):
        clean_img, iso, ratio = self.pair_list[idx]
        darkframe = self.select_random_darkframe(iso)
        darkshading = self.darkshading_dict[iso]

        # --------------
        # clip value range of darkframe
        darkframe = util.quantify_numpy(darkframe)
        # --------------

        darkframe = darkframe - darkshading
        
        if self.args.randomcrop_darkshading:
            clean_img = self.aug([clean_img], self.args.crop_size, self.args.crop_size)[0]
            darkframe = self.aug([darkframe], self.args.crop_size, self.args.crop_size)[0]
        else:
            clean_img, darkframe = self.aug([clean_img, darkframe],
                                 self.args.crop_size, self.args.crop_size)

        # -----------------------
        # add darkshading estimation error simulation
        # darkframe = raw_util.add_darkshading_error(
        #     darkframe, 
        #     std, 
        #     n_frames=400
        # )
        # -----------------------
        darkframe = darkframe * (16383 - 512)
        
        noisy_img = self.apply_noise(clean_img, darkframe, iso, ratio)
        clean_img = clean_img / (16383 - 512)
        noisy_img = noisy_img / (16383 - 512)

        sample = {
                  'clean_img': clean_img,
                  'noisy_img': noisy_img,
                 }

        for key in sample.keys():
            if key not in ['iso', 'ratio', 'iso_ratio_idx', 'noisy_name', 'clean_name']:
                sample[key] = sample[key].astype(np.float32)
                sample[key] = torch.from_numpy(sample[key]).float()

        return sample


# ----------------------------------------------------------------------
# use all clean images for every camera settings
# ----------------------------------------------------------------------
