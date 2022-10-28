import numpy as np
from scipy import ndimage
import torch.utils.data
from skimage import segmentation, morphology, io, measure
import skimage
import random
import torch
import torchvision.transforms.functional as TF

#%%
class PNGNpyLoader(torch.utils.data.Dataset):
    """
    Data Loader. Loads images from a file list  
    compared to V1: check skimage version and add different mode    
    """
    def __init__(
        self,
        img_path, # png format
        ann_path, # numpy format
        img_transform=None, # e.g., ToTensor()
        data_augmentation=True,
        mode='train' # train: apply data augmentation; others: do nothing
    ):
        # assert input_shape is not None and mask_shape is not None
        self.imgs = img_path
        self.anns = ann_path
        self.img_transform = img_transform
        self.data_augmentation = data_augmentation
        self.mode = mode

    def __len__(self):
        return len(self.imgs)

    def my_transform(self, image, interior_cell_tensor, dist_cell_tensor, interior_nuc_tensor, dist_nuc_tensor):
        if random.random()>0.5:
            image = TF.hflip(image)
            interior_cell_tensor = TF.hflip(interior_cell_tensor)
            dist_cell_tensor = TF.hflip(dist_cell_tensor)
            interior_nuc_tensor = TF.hflip(interior_nuc_tensor)
            dist_nuc_tensor = TF.hflip(dist_nuc_tensor)            
        if random.random()>0.5:
            image = TF.vflip(image)
            interior_cell_tensor = TF.vflip(interior_cell_tensor)
            dist_cell_tensor = TF.vflip(dist_cell_tensor)
            interior_nuc_tensor = TF.vflip(interior_nuc_tensor)
            dist_nuc_tensor = TF.vflip(dist_nuc_tensor)    
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
        return image, interior_cell_tensor, dist_cell_tensor, interior_nuc_tensor, dist_nuc_tensor

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
    
    def __getitem__(self, idx):
        # Read RGB images
        img = io.imread(self.imgs[idx])# .astype(np.float32)

        if self.img_transform:
            img_tensor = self.img_transform(img)
        else:
            img = img/np.max(img)
            img_tensor = torch.from_numpy(img).permute(2,0,1)

        # whole-cell instance map 
        inst_map = np.load(self.anns[idx], allow_pickle=True) #(H,W,2) 0-cell; 1-nuclei
        inst_map = morphology.remove_small_objects(inst_map, min_size=15)
        # cell interior and distance map
        inst_map_cell, _, _ = segmentation.relabel_sequential(inst_map[...,0])
        interior_cell = self.create_interior_map(inst_map_cell)
        dist_cell = self.create_distance_map(interior_cell)
        
        inst_map_nuc, _, _ = segmentation.relabel_sequential(inst_map[...,1])
        # nuclear interior and distance map
        interior_nuc = self.create_interior_map(inst_map_nuc)
        dist_nuc = self.create_distance_map(interior_nuc)
    
        # convert numpy array to torch tensor
        interior_cell_tensor = torch.from_numpy(np.expand_dims(interior_cell, 0))
        dist_cell_tensor = torch.from_numpy(np.expand_dims(dist_cell, 0))
        interior_nuc_tensor = torch.from_numpy(np.expand_dims(interior_nuc, 0))
        dist_nuc_tensor = torch.from_numpy(np.expand_dims(dist_nuc, 0))

        if self.mode == 'train':
            if self.data_augmentation:
                img_tensor, interior_cell_tensor, dist_cell_tensor, interior_nuc_tensor, dist_nuc_tensor = self.my_transform(img_tensor, interior_cell_tensor, dist_cell_tensor, interior_nuc_tensor, dist_nuc_tensor)  
            feed_dict = {'img': img_tensor, 'interior_cell':interior_cell_tensor, 'dist_cell':dist_cell_tensor, 
                         'interior_nuc':interior_nuc_tensor, 'dist_nuc':dist_nuc_tensor, 'name':self.imgs[idx]}
        elif self.mode == 'validation':
            inst_map_relabel = np.zeros_like(inst_map, dtype=np.int16)
            inst_map_relabel[:,:,0] = inst_map_cell
            inst_map_relabel[:,:,1] = inst_map_nuc
            inst_map_tensor = torch.from_numpy(inst_map_relabel)
            feed_dict = {'img': img_tensor, 'interior_cell':interior_cell_tensor, 'dist_cell':dist_cell_tensor, 
                         'interior_nuc':interior_nuc_tensor, 'dist_nuc':dist_nuc_tensor,
                         'inst':inst_map_tensor, 'name':self.imgs[idx]}            
        else:
            feed_dict = {'img': img_tensor}

        return feed_dict














