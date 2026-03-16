import os
import numpy as np
import glob
import random
import math
import pickle
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import rawpy
import scipy
from scipy.stats import truncnorm
import sys
sys.path.append('..')
from utils import util
from utils import raw_util


train_path = "/scratch/students/2023-fall-sp-liying/dataset/SID/Sony_train_list.txt"
sid_folder = "/scratch/students/2023-fall-sp-liying/dataset/SID"
resource_folder = './resources'
lrid_folder = '/scratch/students/2023-fall-sp-liying/dataset/LRID/'


# -----------------------------------------
# SID
# Synthetic dark frames + Poisson
# -----------------------------------------
class SIDSyntheticDataset(Dataset):
    def __init__(self, args):
        self.args = args
        iso_value = args.iso_value
        ratio_value = args.ratio_value
        
        pair_list = []

        # get paths of dark frame files
        self.darkframe_dict = self.load_darkframe_paths()
        iso_available = list(self.darkframe_dict.keys())
        self.iso_available = iso_available
        
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
                    
                    in_path = os.path.join(sid_folder, in_path)
                    gt_path = os.path.join(sid_folder, gt_path)

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
        noise_profile_path = os.path.join(resource_folder, 'sid_noise_profile_from_single_noisy.pkl')

        with open(noise_profile_path, 'rb') as file:
            self.noise_profile = pickle.load(file)

        # load darkshading dict
        self.darkshading_dict = self.load_darkshadings()


    def __len__(self):
        return self.data_len 


    def load_darkshadings(self):
        darkshading_dict = {}
        files = sorted(glob.glob(os.path.join(self.args.darkshading_folder, "*.npy")))
        
        for file_path in files:
            darkshading = np.load(file_path)  # darkshading was normalized to [0,1]
            iso = int(os.path.basename(file_path).split('.')[0].replace('ISO', ''))
            darkshading_dict[iso] = darkshading
            
        return darkshading_dict

        
    def load_darkframe_paths(self):
        files = sorted(glob.glob(os.path.join(self.args.darkframe_folder, "*.npy")))
        darkframe_dict = {}
        for file_path in files:
            iso = int(os.path.basename(file_path).split('_')[0].replace('ISO', ''))
            if iso in darkframe_dict.keys():
                ## Only use [darkframe_num] samples per ISO!!!
                if len(darkframe_dict[iso]) >= self.args.darkframe_num:
                    continue
                darkframe_dict[iso].append(file_path)
            else:
                darkframe_dict[iso] = []

        if self.args.preload_files:
            darkframe_dict_temp = {}
            for iso, path_list in darkframe_dict.items():
                darkframe_dict_temp[iso] = []
                for path in path_list:
                    darkframe_dict_temp[iso].append(np.load(path))
            darkframe_dict = darkframe_dict_temp

        return darkframe_dict


    def select_random_darkframe(self, iso):
        if self.args.preload_files:
            darkframe_list = self.darkframe_dict[iso]
            idx = np.random.choice(np.arange(0, len(darkframe_list)))
            darkframe = darkframe_list[idx]
        else:
            darkframe_path_list = self.darkframe_dict[iso]
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

        latent = clean / float(ratio)
        C, H, W = latent.shape
        latent = latent.reshape(C*H*W)
        k = self.generate_truncated_normal(K, 1, lower_bound=0.7*K, upper_bound=1.3*K, sample_size=1)
        poisson = k * np.random.poisson(latent / k, size=C*H*W).reshape((C,H, W))
        noisy = (poisson + darkframe)
            
        noisy = noisy * ratio
        noisy = noisy.clip(0, 16383 - 512)
        
        return noisy
        

    def __getitem__(self, idx):
        clean_img, iso, ratio = self.pair_list[idx]
        darkframe = self.select_random_darkframe(iso)
        darkshading = self.darkshading_dict[iso]
        darkframe = util.quantify_numpy(darkframe)
        darkframe = darkframe - darkshading
        
        if self.args.randomcrop_darkshading:
            clean_img = self.aug([clean_img], self.args.crop_size, self.args.crop_size)[0]
            darkframe = self.aug([darkframe], self.args.crop_size, self.args.crop_size)[0]
        else:
            clean_img, darkframe = self.aug([clean_img, darkframe],
                                 self.args.crop_size, self.args.crop_size)

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





# -----------------------------------------
# LRID
# Synthetic dark frames + Poisson
# -----------------------------------------
def raw2bayer(raw, wp=1023, bl=64, norm=True, clip=False):
    raw = raw.astype(np.float32)
    H, W = raw.shape
    out = np.stack((raw[0:H:2, 0:W:2], #RGBG
                    raw[0:H:2, 1:W:2],
                    raw[1:H:2, 1:W:2],
                    raw[1:H:2, 0:W:2]), axis=0).astype(np.float32) 
    if norm:
        out = (out - bl) / (wp - bl)
    if clip: out = np.clip(out, 0, 1)
        
    return out.astype(np.float32) 


def load_raw(in_path, wp=1023, bl=64, norm=True, clip=False):
    raw = rawpy.imread(in_path)
    raw = raw.raw_image_visible.astype(np.float32)
    img = raw2bayer(raw, wp, bl, norm, clip)
    return img

    
def apply_gaussian_blur_numpy(inp, sigma=50):
    """
    Apply Gaussian blur to a (C, H, W) image independently per channel.
    """
    out = np.empty_like(inp)
    for c in range(inp.shape[0]):
        out[c] = scipy.ndimage.gaussian_filter(inp[c], sigma=sigma)
    return out
    

class LRIDSyntheticDataset(Dataset):
    def __init__(self, args):
        self.args = args
        iso_value = args.iso_value
        ratio_value = args.ratio_value
        self.black_level = 64
        self.white_level = 1023

        # ------- test set -------
        indoor_x5_ratio_list = [1, 2, 4, 8, 16]
        indoor_x5_scene_list = [4, 14, 25, 41, 44, 51, 52, 53, 58]
        # outdoor_x3
        outdoor_x3_ratio_list = [1, 2, 4]
        outdoor_x3_scene_list = [9, 21, 22, 32, 44, 51]
        # ------------------------
        
        self.root = lrid_folder
        self.condition_folders = ['indoor_x3', 'indoor_x5', 'outdoor_x3']
        noisy_path_list = []
        clean_path_list = []
        ratio_list = []
        hot_list = []
        for folder in self.condition_folders:
            clean_folder = os.path.join(root, folder, 'npy/GT_align_ours')
            ratios = os.listdir(os.path.join(root, folder, '6400'))
            for ratio in ratios:
                scenes = sorted(os.listdir(os.path.join(root, folder, '6400', ratio)))
                for scene in scenes:
                    if folder == 'indoor_x5' and int(ratio) in indoor_x5_ratio_list and int(scene) in indoor_x5_scene_list:
                        continue
                    if folder == 'outdoor_x3' and int(ratio) in outdoor_x3_ratio_list and int(scene) in outdoor_x3_scene_list:
                        continue
                        
                    noisy_paths = sorted(glob.glob(os.path.join(root, folder, '6400', ratio, scene, '*.dng')))
                    clean_paths = [os.path.join(root, clean_folder, scene+'.npy')] * len(noisy_paths)
                    noisy_path_list.extend(noisy_paths)
                    clean_path_list.extend(clean_paths)
                    ratio_list.extend([float(ratio)] * len(noisy_paths))
                    hot = raw_util.hot_check(folder, int(scene))
                    hot_list.extend([hot] * len(noisy_paths))


        self.pair_list = list(zip(clean_path_list, noisy_path_list, ratio_list, hot_list))
        print('sample number: ', len(self.pair_list))
        
        # read noise profile
        with open(os.path.join(resource_folder, 'lrid_noise_profile_from_single_noisy.pkl'), 'rb') as file:
            self.noise_profile = pickle.load(file)

        # load darkshadings
        self.load_all_darkshadings()
        self.load_all_darkframe_paths()
        self.load_all_clean_imgs()

    
    def __len__(self):
        return len(self.pair_list)

        
    def load_all_clean_imgs(self):
        clean_img_dict = {}
        for folder in self.condition_folders:
            clean_folder = os.path.join(self.root, folder, 'npy/GT_align_ours')
            clean_paths = sorted(glob.glob(os.path.join(clean_folder, '*.npy')))
            for path in clean_paths:
                clean_img = self.load_clean_img(path)
                clean_img_dict[path] = clean_img
        self.clean_img_dict = clean_img_dict

        
    def load_all_darkshadings(self):
        if self.args.use_realdarkshading:
            darkshading = np.load(self.args.darkshading_folder)
            self.darkshading = raw2bayer(darkshading, norm=False, clip=False) / (self.white_level - self.black_level)

            hot_darkshading = np.load(self.args.hot_darkshading_folder)
            self.hot_darkshading = raw2bayer(hot_darkshading, norm=False, clip=False) / (self.white_level - self.black_level)
        else:
            # Load normal darkshadings
            folder = self.args.darkshading_folder
            paths = sorted(glob.glob(os.path.join(folder, '*.npy')))
            darkshading_dict = {}
            for path in paths:
                name = os.path.basename(path).split('+')[0]
                darkshading_dict[name] = np.load(path)
            self.darkshading_dict = darkshading_dict
        
            # Load hot darkshadings
            if self.args.hot_darkshading_folder:
                folder_hot = self.args.hot_darkshading_folder
                paths_hot = sorted(glob.glob(os.path.join(folder_hot, '*.npy')))
                for path in paths_hot:
                    name = os.path.basename(path).split('+')[0]
                    self.darkshading_dict[name] = np.load(path)

        
    def load_clean_img(self, path):
        clean = np.load(path)
        clean = raw2bayer(clean, wp=self.white_level, bl=self.black_level, norm=True, clip=True)
        return clean

    
    def load_noisy_img(self, path):
        noisy = load_raw(path, wp=self.white_level, bl=self.black_level, norm=True, clip=False)
        return noisy

        
    def load_darkframe(self, path):
        darkframe = np.load(path)
        return darkframe

        
    def load_all_darkframe_paths(self):
        # Load normal darkframe paths
        dark_paths = sorted(glob.glob(os.path.join(self.args.darkframe_folder, '*.npy')))
        self.darkframe_path_list = dark_paths
    
        # Load hot darkframe paths
        if self.args.hot_darkframe_folder:
            hot_paths = sorted(glob.glob(os.path.join(self.args.hot_darkframe_folder, '*.npy')))
            self.hot_darkframe_path_list = hot_paths

    
    def select_random_darkframe(self, hot):
        if hot:
            if self.hot_darkframe_path_list:
                path = np.random.choice(self.hot_darkframe_path_list)
            else:
                path = np.random.choice(self.darkframe_path_list)
        else:
            path = np.random.choice(self.darkframe_path_list)

        name = os.path.basename(path).split('+')[0]
        darkframe = self.load_darkframe(path)
        if self.args.use_realdarkshading:
            if hot:
                darkshading = self.hot_darkshading
            else:
                darkshading = self.darkshading
        else:
            darkshading = self.darkshading_dict[name]
            
        darkframe = darkframe - darkshading
        
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
        latent = clean / float(ratio)
        C, H, W = latent.shape
        latent = latent.reshape(C*H*W)
        k = self.generate_truncated_normal(K, 1, lower_bound=0.7*K, upper_bound=1.3*K, sample_size=1)
        poisson = k * np.random.poisson(latent / k, size=C*H*W).reshape((C,H, W))

        noisy = (poisson + darkframe)
        noisy = noisy * ratio
        
        return noisy

        
    def __getitem__(self, idx):
        clean_path, noisy_path, ratio, hot = self.pair_list[idx]
        clean_img = self.clean_img_dict[clean_path]
        darkframe = self.select_random_darkframe(hot)
            
        if self.args.randomcrop_darkshading:
            clean_img = self.aug([clean_img], self.args.crop_size, self.args.crop_size)[0]
            darkframe = self.aug([darkframe],
                                 self.args.crop_size, self.args.crop_size)[0]
        else:
            clean_img, darkframe = self.aug([clean_img, darkframe],
                                 self.args.crop_size, self.args.crop_size)

        clean_img = clean_img * (self.white_level - self.black_level)
        darkframe = darkframe * (self.white_level - self.black_level)
        noisy_img = self.apply_noise(clean_img, darkframe, 6400, ratio)

        clean_img = clean_img / (self.white_level - self.black_level)
        noisy_img = noisy_img / (self.white_level - self.black_level)
        

        sample = {
                  'clean_img': clean_img,
                  'noisy_img': noisy_img,
                 }

        for key in sample.keys():
            if key not in ['iso', 'ratio', 'iso_ratio_idx', 'noisy_name', 'clean_name']:
                sample[key] = sample[key].astype(np.float32)
                sample[key] = torch.from_numpy(sample[key]).float()

        return sample






