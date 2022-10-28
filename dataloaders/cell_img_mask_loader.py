#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 24 16:35:37 2022

2022.06.04: add random crop
@author: jma
"""

import numpy as np
from scipy import ndimage
import torch.utils.data
from skimage import segmentation, morphology, io, measure, exposure
import skimage
import random
import torch
import torchvision.transforms.functional as TF
import tifffile as tif
import os
join = os.path.join
#%%
class CellImgMaskLoader(torch.utils.data.Dataset):
    """
    Data Loader. Loads images from a file list  
    compared to V1: check skimage version and add different mode    
    """
    def __init__(
        self,
        img_path, # png format
        ann_path, # tif format
        roi_size=(256,256), 
        data_augmentation=True,
        mode='train' # train: apply data augmentation; others: do nothing
    ):
        # assert input_shape is not None and mask_shape is not None
        self.imgs = img_path
        self.anns = ann_path
        self.roi_size = roi_size
        self.data_augmentation = data_augmentation
        self.mode = mode


    def __len__(self):
        return len(self.imgs)

    def my_transform(self, image, interior_cell_tensor, dist_cell_tensor):
        if random.random()>0.5:
            image = TF.hflip(image)
            interior_cell_tensor = TF.hflip(interior_cell_tensor)
            dist_cell_tensor = TF.hflip(dist_cell_tensor)        
        if random.random()>0.5:
            image = TF.vflip(image)
            interior_cell_tensor = TF.vflip(interior_cell_tensor)
            dist_cell_tensor = TF.vflip(dist_cell_tensor)  
        if random.random()>0.25:
            image = TF.autocontrast(image)
        if random.random()>0.25:
            image = TF.gaussian_blur(image, kernel_size=5, sigma=random.uniform(0.5, 1.5))
        if random.random()>0.25:
            image = TF.adjust_brightness(image, brightness_factor=random.uniform(0.7, 1.5))
        if random.random()>0.25:
            image = TF.adjust_contrast(image, contrast_factor=random.uniform(0.7, 1.5))
        if random.random()>0.25:
            image = TF.adjust_gamma(image, gamma=random.uniform(0.7, 1.5))
        if random.random()>0.15:
            image = TF.adjust_hue(image, hue_factor=random.uniform(-0.3,0.3))
        if random.random()>0.25:
            image = TF.adjust_saturation(image, saturation_factor=random.uniform(0.7, 1.5))
        return image, interior_cell_tensor, dist_cell_tensor

    def create_interior_map(self, inst_map):
        """
        Parameters
        ----------
        inst_map : (H,W), np.uint16
            DESCRIPTION.

        Returns
        -------
        interior : (H,W), np.uint8 
            three-class map, values: 0,1,2
            0: background
            1: interior
            2: boundary
        """
        # create interior-edge map
        boundary = segmentation.find_boundaries(inst_map, mode='inner')
        # if skimage.__version__ > '0.18.2':
        #     boundary = morphology.binary_dilation(boundary, footprint=morphology.disk(1))
        # else:
        #     boundary = morphology.binary_dilation(boundary, selem=morphology.disk(1))
        interior_temp = np.logical_and(~boundary, inst_map > 0)
        # interior_temp[boundary] = 0
        interior_temp = morphology.remove_small_objects(interior_temp, min_size=15)
        interior = np.zeros_like(inst_map, dtype=np.uint8)
        interior[interior_temp] = 1
        interior[boundary] = 2
        return interior
    
    def create_distance_map(self, interior):
        # create distance map
        dist_map = ndimage.distance_transform_edt(interior==1)
        # dist_map = dist_map/np.max(dist_map)
        label_matrix = measure.label(interior==1)
        inner_distance = np.zeros_like(dist_map)
        # idea from: https://raw.githubusercontent.com/vanvalenlab/deepcell-tf/master/deepcell/utils/transform_utils.py
        for prop in measure.regionprops(label_matrix, dist_map):
            coords = prop.coords
            center = prop.weighted_centroid
            distance_to_center = np.sum((coords - center) ** 2, axis=1)
            # normalize dist map by region area
            _alpha = 1 / np.sqrt(prop.area)
            center_transform = 1 / (1 + _alpha * distance_to_center)
            coords_x = coords[:, 0]
            coords_y = coords[:, 1]
            inner_distance[coords_x, coords_y] = center_transform
        return inner_distance

    def img_cutoff_rescale(self, img, percentile_low=1, percentile_high=99):
        img_norm = np.zeros_like(img)
        for i in range(img.shape[-1]):
            channel_i = img[:, :, i]
            nonzero_vals = channel_i[np.nonzero(channel_i)]
            if len(nonzero_vals) > 0:
                percentiles = np.percentile(nonzero_vals, [percentile_low, percentile_high])
                rescaled_channel = exposure.rescale_intensity(channel_i, in_range=(percentiles[0], percentiles[1]))
                # get rgb index of current channel
                img_norm[:, :, i] = rescaled_channel        
        return img_norm
    
    def __getitem__(self, idx):
        # Read RGB images
        if self.imgs[idx].endswith('.tif') or self.imgs[idx].endswith('.tiff'):
            img = tif.imread(self.imgs[idx])
        else:    
            img = io.imread(self.imgs[idx])# .astype(np.float32)
        
        assert len(img.shape)<4, "image shape error!"

        if len(img.shape) == 2: # create grey image to three-channel image
            img = np.repeat(np.expand_dims(img, axis=-1), 3, axis=-1)
        if img.shape[-1] > 3: # images with > 3 channels, using the first three channels
            img = img[:,:,0:3]        
        img = self.img_cutoff_rescale(img, percentile_low=1, percentile_high=99)
        H, W, _ = img.shape
        roi_H, roi_W = self.roi_size
        roi_start_H = random.randint(0, H-roi_H)
        roi_start_W = random.randint(0, W-roi_W)
        img_tensor = torch.from_numpy(img[roi_start_H:roi_start_H+roi_H, roi_start_W:roi_start_W+roi_W,:]).permute(2,0,1)

        # cell instance map 
        inst_map = tif.imread(self.anns[idx])
        assert inst_map.shape == (H,W), "ground truth shaper error!" + self.anns[idx]
        inst_map = morphology.remove_small_objects(inst_map, min_size=15)
        # cell interior and distance map
        inst_map_cell, _, _ = segmentation.relabel_sequential(inst_map[roi_start_H:roi_start_H+roi_H, roi_start_W:roi_start_W+roi_W])
        interior_cell = self.create_interior_map(inst_map_cell)
        dist_cell = self.create_distance_map(interior_cell)


        # convert numpy array to torch tensor
        interior_cell_tensor = torch.from_numpy(np.expand_dims(interior_cell, 0))
        dist_cell_tensor = torch.from_numpy(np.expand_dims(dist_cell, 0))


        if self.mode == 'train':
            if self.data_augmentation:
                img_tensor, interior_cell_tensor, dist_cell_tensor = self.my_transform(img_tensor, interior_cell_tensor, dist_cell_tensor)  
            feed_dict = {'img': img_tensor, 'interior_cell':interior_cell_tensor, 'dist_cell':dist_cell_tensor, 
                         'name':self.imgs[idx]}
        elif self.mode == 'validation':
            inst_map_tensor = torch.from_numpy(inst_map_cell.astype(np.int16))
            feed_dict = {'img': img_tensor, 'interior_cell':interior_cell_tensor, 'dist_cell':dist_cell_tensor, 
                         'inst':inst_map_tensor, 'name':self.imgs[idx]}            
        else:
            feed_dict = {'img': img_tensor}

        return feed_dict














