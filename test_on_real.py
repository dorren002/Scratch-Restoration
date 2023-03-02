import argparse
import cv2
import glob
import numpy as np
from collections import OrderedDict
import os
import torch
import requests

from models.network_maswinir import SwinIR as net
from utils import utils_image as util


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='scratch_inpaint')
    parser.add_argument('--scale', type=int, default=1, help='scale factor: 1, 2, 3, 4, 8') # 1 for dn and jpeg car
    parser.add_argument('--training_patch_size', type=int, default=128, help='patch size used in training SwinIR. '
                                       'Just used to differentiate two different settings in Table 2 of the paper. '
                                       'Images are NOT tested patch by patch.')
    parser.add_argument('--large_model', action='store_true', help='use large model, only provided for real image sr')
    parser.add_argument('--model_path', type=str,
                        default='/home/lixuewei/dorren/outdir/inpainting/pconv_inpainting_1120_multishape_defect255/models/650000_G.pth')
    parser.add_argument('--folder_lq', type=str, default='/home/lixuewei/dorren/outdir/real_2x/masked_regular/', help='input low-quality test image folder')
    # parser.add_argument('--folder_lq', type=str, default='/home/lixuewei/dataset/swinir_denoise/test/corrupted_real/', help='input low-quality test image folder')
    parser.add_argument('--folder_mask', type=str, default='/home/lixuewei/dorren/outdir/real_2x/mask_regular/', help='mask for input test image folder')
    parser.add_argument('--folder_gt', type=str, default=None, help='input ground-truth test image folder')
    parser.add_argument('--tile', type=int, default=None, help='Tile size, None for no tile during testing (testing as a whole)')
    parser.add_argument('--tile_overlap', type=int, default=32, help='Overlapping of different tiles')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # set up model
    if os.path.exists(args.model_path):
        print(f'loading model from {args.model_path}')
    else:
        os.makedirs(os.path.dirname(args.model_path), exist_ok=True)
        url = 'https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/{}'.format(os.path.basename(args.model_path))
        r = requests.get(url, allow_redirects=True)
        print(f'downloading model {args.model_path}')
        open(args.model_path, 'wb').write(r.content)
        
    model = define_model(args)
    model.eval()
    model = model.to(device)

    # setup folder and path
    folder_lq, folder_m, save_dir, border, window_size = setup(args)
    os.makedirs(save_dir, exist_ok=True)

    for idx, path in enumerate(sorted(glob.glob(os.path.join(folder_lq, '*')))):
        # read image
        imgname, img_lq, mask = get_image_pair(args, path, folder_m)  # image to HWC-BGR, float32
        # img_lq = np.transpose(img_lq , (2, 0, 1)) 
        # mask = np.transpose(mask, (2, 0, 1))
        # img_lq = torch.from_numpy(img_lq).float().unsqueeze(0).to(device)  
        # mask = torch.from_numpy(mask).float().to(device)
        img_lq = img_lq.to(device)
        mask = mask.to(device)

        # inference
        with torch.no_grad():
            # pad input image to be a multiple of window_size
            # _, _, h_old, w_old = img_lq.size()
            # h_pad = (h_old // window_size + 1) * window_size - h_old
            # w_pad = (w_old // window_size + 1) * window_size - w_old
            # img_lq = torch.cat([img_lq, torch.flip(img_lq, [2])], 2)[:, :, :h_old + h_pad, :]
            # img_lq = torch.cat([img_lq, torch.flip(img_lq, [3])], 3)[:, :, :, :w_old + w_pad]
            # mask = torch.cat([mask, torch.flip(mask, [1])], 1)[:, :h_old + h_pad, :]
            # mask = torch.cat([mask, torch.flip(mask, [2])], 2)[:, :, :w_old + w_pad]
            output = test(img_lq, mask, model, args, window_size)
            # output = output[..., :h_old * args.scale, :w_old * args.scale]

        # save image
        # output = output.data.squeeze().float().cpu().clamp_(0, 1).numpy()
        # if output.ndim == 3:
        #     output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))  # CHW-RGB to HCW-BGR
        # output = (output * 255.0).round().astype(np.uint8)  # float32 to uint8
        output = util.tensor2uint(output)
        print(f'{save_dir}/{imgname}_restored.png')
        util.imsave(output, f'{save_dir}/{imgname}_restored.png')
        # cv2.imwrite(f'{save_dir}/{imgname}_restored.png', output)


def define_model(args):
    model = net(upscale=1, in_chans=3, img_size=128, window_size=8,
                img_range=1., depths=[6, 6, 6, 6, 6, 6], embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6],
                mlp_ratio=2, upsampler='', resi_connection='1conv')
    param_key_g = 'params'
    
    pretrained_model = torch.load(args.model_path)
    model.load_state_dict(pretrained_model[param_key_g] if param_key_g in pretrained_model.keys() else pretrained_model, strict=True)
        
    return model


def setup(args):
    save_dir = f'/home/lixuewei/dorren/outdir/results/pconv_multishape_realimgenmask_255_620k'
    folder_lq = args.folder_lq
    folder_m = args.folder_mask
    border = 0
    window_size = 8

    return folder_lq, folder_m, save_dir, border, window_size


def get_image_pair(args, path, mask_dir):
    (imgname, imgext) = os.path.splitext(os.path.basename(path))   
    img_lq = util.imread_uint(path, 3)
    img_lq = util.single2tensor3(util.uint2single(img_lq)).unsqueeze(0)
    mask = util.imread_uint(os.path.join(mask_dir, imgname+'.png'), 1)
    mask = util.single2tensor2(util.mask_invert(util.uint2single(mask))).unsqueeze(0)
    return imgname, img_lq, mask


def test(img_lq, mask, model, args, window_size):
    if args.tile is None:
        # test the image as a whole
        output = model(img_lq, mask)
    else:
        # test the image tile by tile
        b, c, h, w = img_lq.size()
        tile = min(args.tile, h, w)
        assert tile % window_size == 0, "tile size should be a multiple of window_size"
        tile_overlap = args.tile_overlap
        sf = args.scale

        stride = tile - tile_overlap
        h_idx_list = list(range(0, h-tile, stride)) + [h-tile]
        w_idx_list = list(range(0, w-tile, stride)) + [w-tile]
        E = torch.zeros(b, c, h*sf, w*sf).type_as(img_lq)
        W = torch.zeros_like(E)

        for h_idx in h_idx_list:
            for w_idx in w_idx_list:
                in_patch = img_lq[..., h_idx:h_idx+tile, w_idx:w_idx+tile]
                out_patch = model(in_patch)
                out_patch_mask = torch.ones_like(out_patch)

                E[..., h_idx*sf:(h_idx+tile)*sf, w_idx*sf:(w_idx+tile)*sf].add_(out_patch)
                W[..., h_idx*sf:(h_idx+tile)*sf, w_idx*sf:(w_idx+tile)*sf].add_(out_patch_mask)
        output = E.div_(W)

    return output

if __name__ == '__main__':
    main()
