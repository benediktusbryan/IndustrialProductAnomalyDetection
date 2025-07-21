import os
from PIL import Image
import numpy as np
from functools import reduce
import cv2

import torch
from torch.utils.data import Dataset
import torchvision.transforms as ts

class Base(Dataset):

    def __init__(self, params):
        for key, value in params.items():
            self.__setattr__(key, value)
        mean = self.normalize['mean']
        std = self.normalize['std']
        self.img_trans = ts.Compose(
            [
                ts.Resize(self.out_size),
                ts.ToTensor(),
                ts.Normalize(**self.normalize)
            ]
        )
        self.gt_trans = ts.Compose(
            [
                ts.Resize(self.out_size),
                ts.ToTensor()
            ]
        )
        self.inv_trans = ts.Compose(
            [
                ts.Normalize(
                    mean=[-m / s for m, s in zip(mean, std)], 
                    std=[1. / s for s in std]
                )
            ]
        )
        self.load_data()

    def __len__(self):
        return len(self.img_paths)
    
    def _get_gt(self, idx):
        gt = self.gt_paths[idx]
        return torch.zeros([1, *self.out_size]) if gt == 0 else self.gt_trans(Image.open(gt).convert('L'))
    
    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        img = Image.open(img_path).convert('RGB')
        gt = self._get_gt(idx)
    
        filename = self.filenames[idx]
        
        # Augmentation Process
        if self.mode == 'temp':
            if '_hflip' in filename:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
                if isinstance(gt, Image.Image):
                    gt = gt.transpose(Image.FLIP_LEFT_RIGHT)
            elif '_vflip' in filename:
                img = img.transpose(Image.FLIP_TOP_BOTTOM)
                if isinstance(gt, Image.Image):
                    gt = gt.transpose(Image.FLIP_TOP_BOTTOM)
            elif '_rot180' in filename:
                img = img.rotate(180, expand=True)
                if isinstance(gt, Image.Image):
                    gt = gt.rotate(180, expand=True)
            elif '_rot90' in filename:
                img = img.rotate(90, expand=True)
                if isinstance(gt, Image.Image):
                    gt = gt.rotate(90, expand=True)
            elif '_rot270' in filename:
                img = img.rotate(270, expand=True)
                if isinstance(gt, Image.Image):
                    gt = gt.rotate(270, expand=True)
    
        img = self.img_trans(img)
    
        # Transform gt if still Image not Tensor
        if isinstance(gt, Image.Image):
            gt = self.gt_trans(gt)
    
        assert img.size()[1:] == gt.size()[1:], "image.size != gt.size !!!"
        return img, gt
    
class MVTec_AD(Base):

    def __init__(self, **params):
        super(MVTec_AD, self).__init__(params)

    def load_data(self):
        self.img_path = os.path.join(self.datapath, self.category, self.mode if self.mode == 'test' else 'train')
        if self.mode == 'test':
            self.gt_path = os.path.join(self.datapath, self.category, 'ground_truth')
        self.img_paths, self.gt_paths, self.labels, self.types = [], [], [], []
        for defect_type in filter(lambda x : os.path.isdir(os.path.join(self.img_path, x)), os.listdir(self.img_path)):
            img_paths = [os.path.join(self.img_path, defect_type, x) for x in sorted(filter(lambda x : x.endswith('.png'), \
                os.listdir(os.path.join(self.img_path, defect_type))))]
            gt_paths = [os.path.join(self.gt_path, defect_type, x) for x in sorted(filter(lambda x : x.endswith('.png'), \
                os.listdir(os.path.join(self.gt_path, defect_type))))] if defect_type != 'good' else [0] * len(img_paths)
            self.img_paths.extend(img_paths)
            self.gt_paths.extend(gt_paths)
            self.labels.extend([int(defect_type != 'good')] * len(img_paths))
            self.types.extend([defect_type] * len(img_paths))
        self.filenames = [os.path.basename(reduce(lambda x1, x2 : x1 + '.' + x2, img_path.split('.')[:-1])) for img_path in self.img_paths]

        # Augmentation
        if self.mode == 'temp' and self.category in ['grid', 'hazelnut', 'transistor', 'zipper', 'pill', 'toothbrush', 'carpet', 'wood']:
            print('Augmentation')
            augmented_img_paths = []
            augmented_gt_paths = []
            augmented_labels = []
            augmented_types = []
            augmented_filenames = []
            
            if self.category in ['grid', 'hazelnut', 'transistor', 'zipper']:
                suffix = '_hflip'
            elif self.category in ['toothbrush']:
                suffix = '_rot90'
            elif self.category in ['carpet', 'wood']:
                suffix = '_rot270'
            
            for i in range(len(self.img_paths)):
                augmented_img_paths.append(self.img_paths[i])
                augmented_gt_paths.append(self.gt_paths[i])
                augmented_labels.append(self.labels[i])
                augmented_types.append(self.types[i] + suffix)
                augmented_filenames.append(self.filenames[i] + suffix)
        
            self.img_paths.extend(augmented_img_paths)
            self.gt_paths.extend(augmented_gt_paths)
            self.labels.extend(augmented_labels)
            self.types.extend(augmented_types)
            self.filenames.extend(augmented_filenames)

        assert len(self.img_paths) == len(self.gt_paths), "Something wrong with test and ground truth pair!"
