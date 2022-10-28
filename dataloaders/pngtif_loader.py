#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
add random crop
support loading images from various shapes
"""
import numpy as np
from scipy import ndimage
import torch.utils.data
from skimage import segmentation, morphology, io, measure, exposure
import random
import torch
import torchvision.transforms.functional as TF
import tifffile as tif

class PNGTIFLoader(torch.utils.data.Dataset):
    """
    Data Loader. Loads images from a file list  
    default data normalization
    """


    def __init__(
        self,
        img_path, # png format
        ann_path, # numpy format
        crop_size=256,
        mode='train' # train: apply data augmentation; others: do nothing
    ):
        # assert input_shape is not None and mask_shape is not None
        self.imgs = img_path
        self.anns = ann_path
        self.crop_size = crop_size
        self.mode = mode

    def __len__(self):
        return len(self.imgs)

    def my_transform(self, image, mask, dist):
        if random.random()>0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)
            dist = TF.hflip(dist)
        if random.random()>0.5:
            image = TF.vflip(image)
            mask = TF.vflip(mask)
            dist = TF.vflip(dist)
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
        # if random.random()>0.15:
        #     image = TF.adjust_hue(image, hue_factor=random.uniform(-0.3,0.3))
        # if random.random()>0.25:
        #     image = TF.adjust_saturation(image, saturation_factor=random.uniform(0.7, 1.5))
        return image, mask, dist

    def normalize_channel(self, img, lower=0.1, upper=99.9):
        non_zero_vals = img[np.nonzero(img)]
        percentiles = np.percentile(non_zero_vals, [lower, upper])
        if percentiles[1] - percentiles[0] > 0.001:
            img_norm = exposure.rescale_intensity(img, in_range=(percentiles[0], percentiles[1]), out_range='uint8')
        else:
            img_norm = img
        return img_norm


    def random_crop(self, img, mask, height=256, width=256):
        if img.shape[0] < height:
            pad_hight = np.ceil((height - img.shape[0]) / 2).astype(np.uint8)
            img = np.pad(img, ((pad_hight, pad_hight), (0, 0), (0, 0)), constant_values=0)
            mask = np.pad(mask, ((pad_hight, pad_hight), (0, 0)), constant_values=0) 
        if img.shape[1] < width:
            pad_width = np.ceil((width - img.shape[1]) / 2).astype(np.uint8)
            img = np.pad(img, ((0, 0), (pad_width, pad_width), (0, 0)), constant_values=0)
            mask = np.pad(mask, ((0, 0), (pad_width, pad_width)), constant_values=0)            
        assert img.shape[0] == mask.shape[0]
        assert img.shape[1] == mask.shape[1]
        x = random.randint(0, img.shape[0] - height)
        y = random.randint(0, img.shape[1] - width)
        img = img[x:x+height, y:y+width, :]
        mask = mask[x:x+height, y:y+width]
        return img, mask

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
        # interior_temp = morphology.remove_small_objects(interior_temp, min_size=15)
        interior = np.zeros_like(inst_map, dtype=np.uint8)
        interior[interior_temp] = 1
        interior[boundary] = 2
        return interior

    def __getitem__(self, idx):
        # Read RGB images
        img_name = self.imgs[idx]
        if img_name.endswith('.tif') or img_name.endswith('.tiff'):
            img_data = tif.imread(img_name)
        else:
            img_data = io.imread(img_name)
        
        # normalize image data to 255
        if len(img_data.shape) == 2:
            img_norm = self.normalize_channel(img_data, lower=0.1, upper=99.9)
            pre_img_data = np.repeat(np.expand_dims(img_norm, axis=-1), 3, axis=-1)
        elif len(img_data.shape) == 3:
            pre_img_data = np.zeros((img_data.shape[0], img_data.shape[1], 3), dtype=np.uint8)
            for i in range(3):
                img_channel_i = img_data[:,:,i]
                if len(img_channel_i[np.nonzero(img_channel_i)])>0:
                    pre_img_data[:,:,i] = self.normalize_channel(img_channel_i, lower=0.1, upper=99.9)
        else:
            raise ValueError('image shape should be (H, W) or (H, W, 3)')
        img_whole = pre_img_data/np.max(pre_img_data)

        
        # read tiff mask
        inst_map_whole = np.int16(tif.imread(self.anns[idx]))
        assert len(inst_map_whole.shape) == 2, 'mask shape should be (H, W)'
        if self.mode == 'train':
            img, inst_map = self.random_crop(img_whole, inst_map_whole, width=self.crop_size, height=self.crop_size)
            # inst_map, _, _ = segmentation.relabel_sequential(inst_map)
            while len(np.unique(inst_map))<1:
                img, inst_map = self.random_crop(img_whole, inst_map_whole, width=self.crop_size, height=self.crop_size)
            if np.any(np.isnan(img)):
                print(img_name, 'nan error')

            img_tensor = torch.from_numpy(img).permute(2,0,1)
            
            # create interior-edge map
            # inst_map,_,_ = segmentation.relabel_sequential(inst_map)
            # interior = np.zeros_like(inst_map, dtype=np.uint8)
            # boundary = segmentation.find_boundaries(inst_map, mode='inner')
            # boundary = morphology.binary_dilation(boundary, morphology.disk(1))
            # interior_temp = np.logical_and(boundary == 0, inst_map > 0).astype(np.uint8)
            # # interior_temp[boundary] = 0
            # interior_temp = morphology.remove_small_objects(interior_temp==1)
            # interior[interior_temp==1] = 1
            # interior[boundary==1] = 2
            
            interior = self.create_interior_map(inst_map)
            
        
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
    
            interior_tensor = torch.from_numpy(np.expand_dims(interior, 0))
            inner_distance_tensor = torch.from_numpy(np.expand_dims(inner_distance, 0))        
        
            img_tensor, interior_tensor, inner_distance_tensor = self.my_transform(img_tensor, interior_tensor, inner_distance_tensor)  
            feed_dict = {"img": img_tensor, 'interior':interior_tensor, 'dist':inner_distance_tensor, 'name':self.imgs[idx]}
        elif self.mode == 'validation':
            interior = self.create_interior_map(inst_map_whole)
            
            img_tensor = torch.from_numpy(img_whole).permute(2,0,1)
            inst_map_tensor = torch.from_numpy(inst_map_whole.astype(np.int16))
            interior_tensor = torch.from_numpy(np.expand_dims(interior, 0))
            feed_dict = {"img": img_tensor, 'interior':interior_tensor, 'inst':inst_map_tensor, 'name':self.imgs[idx]}
        else:
            feed_dict = {"img": img_tensor}

        return feed_dict




class PNGTIFLoader2(torch.utils.data.Dataset):
    """
    Data Loader. Loads images from a file list  
    compared to V1: check skimage version and add different mode    
    """


    def __init__(
        self,
        img_path, # png format
        ann_path, # numpy format
        crop_size=256,
        mode='train' # train: apply data augmentation; others: do nothing
    ):
        # assert input_shape is not None and mask_shape is not None
        self.imgs = img_path
        self.anns = ann_path
        self.crop_size = crop_size
        self.mode = mode

    def __len__(self):
        return len(self.imgs)

    def my_transform(self, image, mask, dist):
        if random.random()>0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)
            dist = TF.hflip(dist)
        if random.random()>0.5:
            image = TF.vflip(image)
            mask = TF.vflip(mask)
            dist = TF.vflip(dist)
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
        # if random.random()>0.15:
        #     image = TF.adjust_hue(image, hue_factor=random.uniform(-0.3,0.3))
        # if random.random()>0.25:
        #     image = TF.adjust_saturation(image, saturation_factor=random.uniform(0.7, 1.5))
        return image, mask, dist

    # def normalize_channel(self, img, lower=1, upper=99):
    #     non_zero_vals = img[np.nonzero(img)]
    #     percentiles = np.percentile(non_zero_vals, [lower, upper])
    #     if percentiles[1] - percentiles[0] > 0.001:
    #         img_norm = exposure.rescale_intensity(img, in_range=(percentiles[0], percentiles[1]), out_range='uint8')
    #     else:
    #         img_norm = img
    #     return img_norm

    def normalize_channel01(self, img, lower=1, upper=99):
        non_zero_vals = img[np.nonzero(img)]
        percentiles = np.percentile(non_zero_vals, [lower, upper])
        if percentiles[1] - percentiles[0] > 0.001:
            img_norm = (img-percentiles[0]) / (percentiles[1] - percentiles[0])
        else:
            img_norm = img/np.max(img)
        img_norm[img==0] = 0
        return img_norm

    def random_crop(self, img, mask, height=256, width=256):
        if img.shape[0] < height:
            pad_hight = np.ceil((height - img.shape[0]) / 2).astype(np.uint8)
            img = np.pad(img, ((pad_hight, pad_hight), (0, 0), (0, 0)), constant_values=0)
            mask = np.pad(mask, ((pad_hight, pad_hight), (0, 0)), constant_values=0) 
        if img.shape[1] < width:
            pad_width = np.ceil((width - img.shape[1]) / 2).astype(np.uint8)
            img = np.pad(img, ((0, 0), (pad_width, pad_width), (0, 0)), constant_values=0)
            mask = np.pad(mask, ((0, 0), (pad_width, pad_width)), constant_values=0)            
        assert img.shape[0] == mask.shape[0]
        assert img.shape[1] == mask.shape[1]
        x = random.randint(0, img.shape[0] - height)
        y = random.randint(0, img.shape[1] - width)
        img = img[x:x+height, y:y+width, :]
        mask = mask[x:x+height, y:y+width]
        return img, mask

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

    def __getitem__(self, idx):
        # Read RGB images
        img_name = self.imgs[idx]
        if img_name.endswith('.tif') or img_name.endswith('.tiff'):
            img_data = tif.imread(img_name)
        else:
            img_data = io.imread(img_name)
        
        # normalize image data to 255
        # if len(img_data.shape) == 2:
        #     img_norm = self.normalize_channel(img_data, lower=0.1, upper=99.9)
        #     pre_img_data = np.repeat(np.expand_dims(img_norm, axis=-1), 3, axis=-1)
        # elif len(img_data.shape) == 3:
        #     pre_img_data = np.zeros((img_data.shape[0], img_data.shape[1], 3), dtype=np.uint8)
        #     for i in range(3):
        #         img_channel_i = img_data[:,:,i]
        #         if len(img_channel_i[np.nonzero(img_channel_i)])>0:
        #             pre_img_data[:,:,i] = self.normalize_channel(img_channel_i, lower=0.1, upper=99.9)
        # else:
        #     raise ValueError('image shape should be (H, W) or (H, W, 3)')
        # img_whole = pre_img_data/np.max(pre_img_data)

        # normalize image data to 0-1
        if len(img_data.shape) == 2:
            img_norm = self.normalize_channel01(img_data, lower=0.1, upper=99.9)
            pre_img_data = np.repeat(np.expand_dims(img_norm, axis=-1), 3, axis=-1)
        elif len(img_data.shape) == 3:
            pre_img_data = np.zeros((img_data.shape[0], img_data.shape[1], 3))
            for i in range(3):
                img_channel_i = img_data[:,:,i]
                if np.max(img_channel_i)>0.1:
                    pre_img_data[:,:,i] = self.normalize_channel01(img_channel_i, lower=0.1, upper=99.9)
        else:
            raise ValueError('image shape should be (H, W) or (H, W, 3)')
        img_whole = pre_img_data
        
        # read tiff mask
        inst_map_whole = tif.imread(self.anns[idx])
        assert len(inst_map_whole.shape) == 2, 'mask shape should be (H, W)'
        if self.mode == 'train':
            img, inst_map = self.random_crop(img_whole, inst_map_whole, width=self.crop_size, height=self.crop_size)
            inst_map, _, _ = segmentation.relabel_sequential(inst_map)
            while np.max(inst_map)<2:
                img, inst_map = self.random_crop(img_whole, inst_map_whole, width=self.crop_size, height=self.crop_size)
            if np.any(np.isnan(img)):
                print(img_name, 'nan error')

            img_tensor = torch.from_numpy(img).permute(2,0,1)
            
            # create interior-edge map
            # inst_map,_,_ = segmentation.relabel_sequential(inst_map)
            # interior = np.zeros_like(inst_map, dtype=np.uint8)
            # boundary = segmentation.find_boundaries(inst_map, mode='inner')
            # boundary = morphology.binary_dilation(boundary, morphology.disk(1))
            # interior_temp = np.logical_and(boundary == 0, inst_map > 0).astype(np.uint8)
            # # interior_temp[boundary] = 0
            # interior_temp = morphology.remove_small_objects(interior_temp==1)
            # interior[interior_temp==1] = 1
            # interior[boundary==1] = 2
            
            interior = self.create_interior_map(inst_map)
            
        
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
    
            interior_tensor = torch.from_numpy(np.expand_dims(interior, 0))
            inner_distance_tensor = torch.from_numpy(np.expand_dims(inner_distance, 0))        
        
            img_tensor, interior_tensor, inner_distance_tensor = self.my_transform(img_tensor, interior_tensor, inner_distance_tensor)  
            feed_dict = {"img": img_tensor, 'interior':interior_tensor, 'dist':inner_distance_tensor, 'name':self.imgs[idx]}
        elif self.mode == 'validation':
            interior = self.create_interior_map(inst_map_whole)
            
            img_tensor = torch.from_numpy(img_whole).permute(2,0,1)
            inst_map_tensor = torch.from_numpy(inst_map_whole.astype(np.int16))
            interior_tensor = torch.from_numpy(np.expand_dims(interior, 0))
            feed_dict = {"img": img_tensor, 'interior':interior_tensor, 'inst':inst_map_tensor, 'name':self.imgs[idx]}
        else:
            feed_dict = {"img": img_tensor}

        return feed_dict










