def define_Dataset(dataset_opt):
    dataset_type = dataset_opt['dataset_type'].lower()

    if dataset_type in ['dymask']:
        from .dataset_dyMask import DyMaskDataset as D
    elif dataset_type in['denoising']:
        from .dataset_dncnn import DatasetDnCNN as D

    dataset = D(dataset_opt)

    print('Dataset [{:s} - {:s}] is created.'.format(
        dataset.__class__.__name__, dataset_opt['name']))
        
    return dataset
