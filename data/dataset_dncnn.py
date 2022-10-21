import os.path
import random
import numpy as np
import torch
import torch.utils.data as data
import utils.utils_image as util


class DatasetDnCNN(data.Dataset):
    """
    # -----------------------------------------
    # Get L/H for scratch restoration
    # -----------------------------------------
    """

    def __init__(self, opt):
        super(DatasetDnCNN, self).__init__()
        self.opt = opt
        self.n_channels = opt['n_channels'] if opt['n_channels'] else 3
        self.patch_size = opt['H_size'] if opt['H_size'] else 64

        # ------------------------------------
        # get path of H
        # return None if input is None
        # ------------------------------------
        self.paths_H = util.get_image_paths(opt['dataroot_H'])
        self.paths_L = util.get_image_paths(opt['dataroot_L'])

    def __getitem__(self, index):

        # ------------------------------------
        # get H image
        # ------------------------------------
        H_path = self.paths_H[index]
        img_H = util.imread_uint(H_path, self.n_channels)

        L_path = self.paths_L[index]
        img_L = util.imread_uint(L_path, self.n_channels)

        if self.opt['phase'] == 'train':
            """
            # --------------------------------
            # get L/H patch pairs
            # --------------------------------
            """
            H, W, _ = img_H.shape

            # --------------------------------
            # HWC to CHW, numpy(uint) to tensor
            # --------------------------------
            img_H = util.uint2tensor3(img_H)
            img_L = util.uint2tensor3(img_L)

        else:
            """
            # --------------------------------
            # get L/H image pairs
            # --------------------------------
            """
            img_H = util.uint2single(img_H)
            img_L = util.uint2single(img_L)

            # --------------------------------
            # HWC to CHW, numpy to tensor
            # --------------------------------
            img_L = util.single2tensor3(img_L)
            img_H = util.single2tensor3(img_H)

        return {'L': img_L, 'H': img_H, 'H_path': H_path, 'L_path': L_path}

    def __len__(self):
        return len(self.paths_H)
