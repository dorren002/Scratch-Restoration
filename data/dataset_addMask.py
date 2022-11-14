import os.path
import random
import numpy as np
import torch
import torch.utils.data as data
import utils.utils_image as util
import cv2


class DyMaskDataset(data.Dataset):
    """
    # -----------------------------------------
    # Get L/H for scratch restoration
    # -----------------------------------------
    """

    def __init__(self, opt):
        super(DyMaskDataset, self).__init__()
        self.opt = opt
        self.n_channels = opt['n_channels'] if opt['n_channels'] else 3
        self.patch_size = opt['H_size'] if opt['H_size'] else 64

        # ------------------------------------
        # get path of H
        # return None if input is None
        # ------------------------------------
        self.paths_H = util.get_image_paths(opt['dataroot_H'])
        self.paths_L = util.get_image_paths(opt['dataroot_L'])
        self.paths_M = util.get_image_paths(opt['dataroot_M'])

    def gen_mask(self):
        mask = np.zeros((self.patch_size, self.patch_size), dtype='float32')
        rnd_ss = random.randint(5,10)
        for i in range(rnd_ss):
            rnd_mw = random.randint(0, self.patch_size)
            rnd_mh = random.randint(0, self.patch_size)
            rnd_ms = random.randint(10,30)
            mask[rnd_mw:rnd_mw+rnd_ms, rnd_mh:rnd_mh+rnd_ms] = 1
        return mask
    def add_mask(self, img, mask):
        arg_mask = np.argwhere(mask==1)
        for arg in arg_mask:
            x,y = arg
            img[x][y][:] = 255
        return img

    def __getitem__(self, index):

        # ------------------------------------
        # get H image
        # ------------------------------------
        H_path = self.paths_H[index]
        L_path = H_path
        img_H = util.imread_uint(H_path, self.n_channels)

        if self.opt['phase'] == 'train':
            """
            # --------------------------------
            # get L/H patch pairs
            # --------------------------------
            """
            H, W, _ = img_H.shape

            # --------------------------------
            # randomly crop the patch
            # --------------------------------
            rnd_h = random.randint(0, max(0, H - self.patch_size))
            rnd_w = random.randint(0, max(0, W - self.patch_size))
            patch_H = img_H[rnd_h:rnd_h + self.patch_size, rnd_w:rnd_w + self.patch_size, :]

            # --------------------------------
            # augmentation - flip, rotate
            # --------------------------------
            mode = random.randint(0, 7)
            patch_H = util.augment_img(patch_H, mode=mode)

            # --------------------------------
            # add mask
            # --------------------------------
            mask = self.gen_mask()

            noise = 15*np.random.standard_normal(patch_H.shape)
            img_L = patch_H + noise
            img_L = self.add_mask(img_L, mask)

            # cv2.imwrite('test.jpg', img_L)
            
            # --------------------------------
            # HWC to CHW, numpy(uint) to tensor
            # --------------------------------
            img_L = util.uint2tensor3(img_L)
            img_H = util.uint2tensor3(patch_H)
            
            mask = util.mask_invert(mask)

        else:
            """
            # --------------------------------
            # get L/H image pairs
            # --------------------------------
            """
            img_H = util.uint2single(img_H)

            L_path = self.paths_L[index]
            img_L = util.imread_uint(L_path, self.n_channels)
            img_L = util.uint2single(img_L)

            # --------------------------------
            # HWC to CHW, numpy to tensor
            # --------------------------------
            img_L = util.single2tensor3(img_L)
            img_H = util.single2tensor3(img_H)
            
            M_path = self.paths_M[index]
            mask = util.imread_uint(M_path, 1)
            mask = util.single2tensor2(util.mask_invert(util.uint2single(mask)))
        return {'L': img_L, 'H': img_H, 'M': mask,  'H_path': H_path, 'L_path': L_path}

    def __len__(self):
        return len(self.paths_H)
