import cv2,random
import numpy as np

def random_line_mask(mask, max_mask_num):
    ''' generate lines like defect mask '''

    h, w = mask.shape
    for i in range(max_mask_num):
        rnd_mw = random.randint(0, w) # mask pos
        rnd_mh = random.randint(0, h) # mask pos
        rnd_lw = random.randint(0, 5) # max line width = 5
        rnd_lh = random.randint(10, h) # max line height = h
        mask[rnd_mw:rnd_mw+rnd_lh, rnd_mh:rnd_mh+rnd_lw] = 1
    return mask

def random_dot_mask(mask, max_mask_num, max_mask_size, min_mask_size):
    ''' generate dots like defect mask '''

    h, w = mask.shape
    for i in range(max_mask_num):
        rnd_mw = random.randint(0, w) # mask pos
        rnd_mh = random.randint(0, h) # mask pos
        rnd_dr = random.randint(min_mask_size, max_mask_size) # dots' radius
        X, Y = np.ogrid[:h, :w]
        dist_from_center = (X - rnd_mh)**2 + (Y - rnd_mw)**2
        args = np.argwhere(dist_from_center<rnd_dr)
        for arg in args:
            mask[arg[0], arg[1]] = 1
    return mask

def random_square_mask(mask, max_mask_num, max_mask_size, min_mask_size):
    ''' generate square like defect mask'''

    h, w = mask.shape
    for i in range(max_mask_num):
        rnd_mw = random.randint(0, h) # mask pos
        rnd_mh = random.randint(0, w) # mask pos
        rnd_sw = random.randint(min_mask_size, max_mask_size) # square width
        rnd_sh = random.randint(min_mask_size, max_mask_size) # square height
        mask[rnd_mw:rnd_mw+rnd_sw, rnd_mh:rnd_mh+rnd_sh] = 1
    return mask

def random_irregular_mask(mask, mask_dir="/home/lixuewei/dorren/outdir/real_2x/irregular_mask_128/", max_mask_idx=3285):
    ''' generate irregular defect mask '''

    if mask_dir is None:
        return
    n = random.randint(0, max_mask_idx)
    mask = cv2.imread(mask_dir + '{}.png'.format(n), 0)
    return mask/255

def add_mask(img, mask, value):
    '''
    add mask to the img
    '''

    arg_mask = np.argwhere(mask==1)
    masked_img = np.copy(img)
    for arg in arg_mask:
        x, y = arg
        masked_img[x][y][:] = value
    return masked_img

def mask_invert(mask):
    return 1-mask

def gen_and_add_mask(image, mask_type=0, max_mask_num=10, max_mask_size=30, min_mask_size=10, defect_value=255, mask_dir = "/home/lixuewei/dorren/outdir/real_2x/irregular_mask_128/"):
    '''
    generate randomly defection mask for image and add to the image
    :param: image [ndarray shape=(c,h,w)] image
    :param: mask_type the shape of defect shape, 0 means None, 1 for line, 2 for dot, 3 for square and 4 for irregular
    :param: max_mask_num the max num for mask number, default=10
    :param: max_mask_size the max size for mask, default=30 pixel
    :param: defect_value value of defect area, in [0,255], default=255
    
    :return: masked_image [ndarray shape=(c,h,w)] image
    :return: mask [ndarray shape=(h,w)] binary mask
    '''

    h, w, _ = image.shape
    mask = np.zeros((h,w), dtype='float32')

    if mask_type==1:
        mask = random_line_mask(mask, max_mask_num)
    elif mask_type==2:
        mask = random_dot_mask(mask, max_mask_num, max_mask_size, min_mask_size)
    elif mask_type==3:
        mask = random_square_mask(mask, max_mask_num, max_mask_size, min_mask_size)
    elif mask_type==4:
        mask = random_irregular_mask(mask, mask_dir)
    
    masked_image = add_mask(image, mask, defect_value)

    return masked_image, mask_invert(mask)