#!/usr/bin/env python3
# encoding: utf-8
# Code modified from https://github.com/Wangyixinxin/ACN
import glob
import os
import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset, DataLoader
import random
from typing import List, Tuple, Optional
import json

# --------------------------------------------------------------
# --------------- brats_divide_pair_loop_fix.py ----------------
import json, os, itertools
from typing import Optional, Tuple, List
import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset

MODES = ["flair", "t1", "t1ce", "t2"]

MASKS_TEST = [[False, False, False, True], [False, True, False, False], [False, False, True, False], [True, False, False, False],
         [False, True, False, True], [False, True, True, False], [True, False, True, False], [False, False, True, True], [True, False, False, True], [True, True, False, False],
         [True, True, True, False], [True, False, True, True], [True, True, False, True], [False, True, True, True],
         [True, True, True, True]]
FULL_ID = 14


class Brats2018_divide(Dataset):
    """
    train=True  : 返回 (partial_sample, full_sample)   – full_sample 顺序循环
    train=False : 返回 单样本
    """

    def __init__(self, json_path, data_root, crop_size=[160, 192, 128], modes=['flair', 't1', 't1ce', 't2'] , train=True, normalization = True, dataset='brats', pretrain=False, isbarebone=False):
        self.data_root = data_root
        self.modes = modes
        self.train = train
        self.crop_size = crop_size
        self.normalization = normalization
        self.dataset = dataset
        self.pretrain = pretrain

        with open(json_path, 'r') as f:
            self.meta = json.load(f)

        # 划分索引
        self.full_idx = [i for i, e in enumerate(self.meta) if e['mask_id'] == FULL_ID]
        # self.part_idx = [i for i, e in enumerate(self.meta) if e['mask_id'] != FULL_ID]
        self.part_idx = [i for i, e in enumerate(self.meta)] # 遍历所有的模态组合

        if isbarebone:
            self.part_idx = self.part_idx + self.full_idx

        if self.train and not self.full_idx:
            raise RuntimeError("训练需要至少一个全模态样本！")

        # —— 固定顺序循环：不再 shuffle ——
        self.full_cycle = itertools.cycle(self.full_idx)

    def _load_item(self, idx: int):
        patient_dir   = self.meta[idx]
        patient_id, mask_id = patient_dir['id'], patient_dir['mask_id']
        mask_ = MASKS_TEST[mask_id]             # True/False * 4

        volumes = []
        modes = list(self.modes) + ['seg']
        p = "_" if self.dataset == 'brats' else '-'
        h = ".nii" if self.dataset == 'brats' else '.nii.gz'
        for mode in modes:
            volume_path = os.path.join(self.data_root, patient_id, patient_id + p + mode + h)
            volume = nib.load(volume_path).get_fdata()
            if not mode == "seg" and self.normalization:
                volume = self.normlize(volume)  # [0, 1.0]
            volumes.append(volume)                  # [h, w, d]
        seg_volume = volumes[-1]
        volumes = volumes[:-1]
        volume, seg_volume = self.aug_sample(volumes, seg_volume)
        # ed_volume = (seg_volume == 2) # peritumoral edema ED
        # net_volume = (seg_volume == 1) # enhancing tumor core NET
        # et_volume = (seg_volume == 4) # enhancing tumor ET
        # bg_volume = (seg_volume == 0)
        
        # seg_volume = [ed_volume, net_volume, et_volume, bg_volume]
        # seg_volume = np.concatenate(seg_volume, axis=0).astype("float32")

        # return (torch.tensor(volume.copy(), dtype=torch.float),
        #         torch.tensor(seg_volume.copy(), dtype=torch.float))
        if self.dataset == 'fets':
            net_volume = ((seg_volume ==1)).astype('uint8')# peritumoral edema ED
            snfh_volume = ((seg_volume == 2)).astype('uint8') # enhancing tumor core NET
            et_volume = ((seg_volume == 4)).astype('uint8') # enhancing tumor ET
            bg_volume = ((seg_volume == 0)).astype('uint8')
        elif self.dataset == 'brats':
            net_volume = ((seg_volume ==1)).astype('uint8')# peritumoral edema ED
            snfh_volume = ((seg_volume == 2)).astype('uint8') # enhancing tumor core NET
            et_volume = ((seg_volume == 4)).astype('uint8') # enhancing tumor ET
            bg_volume = ((seg_volume == 0)|(seg_volume == 3)).astype('uint8')
        elif self.dataset == 'brats24' or self.dataset=='retreat':
            net_volume = ((seg_volume ==1)).astype('uint8')# peritumoral edema ED
            snfh_volume = ((seg_volume == 2)).astype('uint8') # enhancing tumor core NET
            et_volume = ((seg_volume == 3)).astype('uint8') # enhancing tumor ET
            bg_volume = ((seg_volume == 0)|(seg_volume == 4)).astype('uint8')

        seg_volume = [net_volume, snfh_volume, et_volume, bg_volume]
        seg_volume = np.concatenate(seg_volume, axis=0).astype("float32")

        # ---------- cls：各类是否存在 ----------
        cls = torch.tensor(
            [net_volume.any(), snfh_volume.any(), et_volume.any()],
            dtype=torch.float32
        )  # shape == (3,)

        # ---------- 拼接掩码通道 ----------
        mask_tensor = torch.tensor(mask_, dtype=torch.float32).view(4, 1, 1, 1)   # [4]

        mask_volume = np.concatenate(
            [net_volume, snfh_volume, et_volume, bg_volume],
            axis=0
        ).astype(np.float32)  # shape (4, H, W, D)

        return (
            torch.tensor(volume.copy(),  dtype=torch.float32) * mask_tensor,
            torch.tensor(mask_volume.copy(), dtype=torch.float32),
            cls,
            mask_tensor,
            mask_                                          # ✅
        )

    # ---------- Dataset API ----------
    def __len__(self):
        if self.pretrain:
            return len(self.full_idx)
        return len(self.part_idx) if self.train else len(self.meta)

    def __getitem__(self, idx):
        if self.train:
            partial = self._load_item(self.part_idx[idx])
            full    = self._load_item(next(self.full_cycle))  # 顺序取，循环
            return partial, full
        else:
            return self._load_item(idx)

    def aug_sample(self, volumes, mask):
        """
            Args:
                volumes: list of array, [h, w, d]
                mask: array [h, w, d], segmentation volume
            Ret: x, y: [channel, h, w, d]

        """
        x = np.stack(volumes, axis=0)       # [N, H, W, D]
        y = np.expand_dims(mask, axis=0)    # [channel, h, w, d]

        if self.train:
            # crop volume
            x, y = self.random_crop(x, y)
            if random.random() < 0.5:
                x = np.flip(x, axis=1)
                y = np.flip(y, axis=1)
            if random.random() < 0.5:
                x = np.flip(x, axis=2)
                y = np.flip(y, axis=2)
            if random.random() < 0.5:
                x = np.flip(x, axis=3)
                y = np.flip(y, axis=3)
        else:
            x, y = self.center_crop(x, y)

        return x, y

    def random_crop(self, x, y):
        """
        Args:
            x: 4d array, [channel, h, w, d]
        """
        crop_size = self.crop_size
        height, width, depth = x.shape[-3:]
        sx = random.randint(0, height - crop_size[0] - 1)
        sy = random.randint(0, width - crop_size[1] - 1)
        sz = random.randint(0, depth - crop_size[2] - 1)
        crop_volume = x[:, sx:sx + crop_size[0], sy:sy + crop_size[1], sz:sz + crop_size[2]]
        crop_seg = y[:, sx:sx + crop_size[0], sy:sy + crop_size[1], sz:sz + crop_size[2]]

        return crop_volume, crop_seg

    def center_crop(self, x, y):
        crop_size = self.crop_size
        height, width, depth = x.shape[-3:]
        sx = (height - crop_size[0] - 1) // 2
        sy = (width - crop_size[1] - 1) // 2
        sz = (depth - crop_size[2] - 1) // 2
        crop_volume = x[:, sx:sx + crop_size[0], sy:sy + crop_size[1], sz:sz + crop_size[2]]
        crop_seg = y[:, sx:sx + crop_size[0], sy:sy + crop_size[1], sz:sz + crop_size[2]]

        return crop_volume, crop_seg

    def normlize(self, x):
        return (x - x.min()) / (x.max() - x.min())
    
    
    def normlize_brain(self, x, epsilon=1e-8):
        average        = x[np.nonzero(x)].mean()
        std            = x[np.nonzero(x)].std() + epsilon
        mask           = x>0
        sub_mean       = np.where(mask, x-average, x)
        x_normalized   = np.where(mask, sub_mean/std, x)
        return x_normalized

def split_dataset(data_root, test_p):
    patients_dir = glob.glob(os.path.join(data_root, "*GG", "Brats18*"))
    patients_dir.sort()
    N = int(len(patients_dir)*test_p)
    train_patients_list =  patients_dir[N:]
    val_patients_list   =  patients_dir[:N]

    return train_patients_list, val_patients_list

def make_data_loaders_divide(config, dataset='brats'):
    """
    读取 2 个 JSON（train / val），并构造 DataLoader。
    必要的 config 键：
        path_to_data   : BraTS2018 解压目录（每个病例一个子目录）
        path_to_meta   : 放置 Brats2018_train.json / Brats2018_val.json 的目录
        inputshape     : [D, H, W]
        modalities     : ["flair","t1","t1ce","t2"]
        batch_size_tr  : int
        batch_size_va  : int
        dataset        : "brats" or "fets"
    """
    data_root   = config['path_to_data']
    # meta_dir    = config['path_to_meta']          # 新增
    train_json  = config['path_to_train'] 
    val_json    = config['path_to_val'] 
    test_json    = config['path_to_test'] 

    crop_size   = tuple(config['inputshape'])     # e.g. (160,192,128)

    train_ds = Brats2018_divide(
        json_path  = train_json,
        data_root  = data_root,
        crop_size  = crop_size,
        modes      = config['modalities'],
        train      = True,
        dataset    = dataset)

    val_ds = Brats2018_divide(
        json_path  = val_json,
        data_root  = data_root,
        crop_size  = crop_size,
        modes      = config['modalities'],
        train      = False,
        dataset    = dataset)

    test_ds = Brats2018_divide(
        json_path  = test_json,
        data_root  = data_root,
        crop_size  = crop_size,
        modes      = config['modalities'],
        train      = False,
        dataset    = dataset)

    pretrain_ds = Brats2018_divide(
        json_path  = train_json,
        data_root  = data_root,
        crop_size  = crop_size,
        modes      = config['modalities'],
        pretrain   = True,
        dataset    = dataset)
    
    loaders = {
        'train': DataLoader(train_ds,
                            batch_size = int(config['batch_size_tr']),
                            shuffle     = True,
                            num_workers = 4,
                            pin_memory  = True),
        'eval' : DataLoader(val_ds,
                            batch_size = int(config['batch_size_va']),
                            shuffle     = False,
                            num_workers = 4,
                            pin_memory  = True),
        'test' : DataLoader(test_ds,
                            batch_size = int(config['batch_size_va']),
                            shuffle     = False,
                            num_workers = 4,
                            pin_memory  = True),
        'pretrain' : DataLoader(pretrain_ds,
                            batch_size = int(config['batch_size_va']),
                            shuffle     = False,
                            num_workers = 4,
                            pin_memory  = True)
    }
    return loaders

