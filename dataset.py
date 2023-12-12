# Custom dataset
from PIL import Image
import torch.utils.data as data
import numpy as np
import os
import random
import glob


class DatasetFromFolder(data.Dataset):
    def __init__(self, data_dir, img_type, transform=None,
                 resize_h=None, resize_w=None, crop_h=None, crop_w=None, fliplr=False):
        super(DatasetFromFolder, self).__init__()

        self.data_list_ir = sorted(glob.glob(data_dir + "/IR/*." + img_type), key=lambda name: int(name[-9:-5]))
        self.data_list_vi = sorted(glob.glob(data_dir + "/VI/*." + img_type), key=lambda name: int(name[-9:-5]))
        self.transform = transform
        self.resize_h = resize_h
        self.resize_w = resize_w
        self.crop_h = crop_h
        self.crop_w = crop_w
        self.fliplr = fliplr

    def __getitem__(self, index):
        # Load Image
        img_path_ir =  self.data_list_ir[index]
        img_path_vi =  self.data_list_vi[index]
        imgs = [Image.open(img_path_ir), Image.open(img_path_vi)]

        for l in range(len(imgs)):
            img = np.array(imgs[l])
            if len(img.shape) == 2:
                img = img[:,:, np.newaxis]
                img = np.concatenate([img] * 3, 2)
            imgs[l] = Image.fromarray(img.astype('uint8'))

        if self.transform is not None:
            imgs = [self.transform(img) for img in imgs]

        return imgs

    def __len__(self):
        return len(self.data_list_ir)
    
