import numpy as np
from scipy import ndimage
import torch.utils.data
from skimage import segmentation, morphology, io, measure
import random
import torch
import torchvision.transforms.functional as TF


class PNGNpyLoader(torch.utils.data.Dataset):
    """
    Data Loader. Loads images from a file list      
    """


    def __init__(
        self,
        img_path, # png format
        ann_path, # numpy format
        img_transform=None,
        mode='test' # train: apply data augmentation; others: do nothing
    ):
        # assert input_shape is not None and mask_shape is not None
        self.imgs = img_path
        self.anns = ann_path
        self.img_transform = img_transform
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
        if random.random()>0.15:
            image = TF.adjust_hue(image, hue_factor=random.uniform(-0.3,0.3))
        if random.random()>0.25:
            image = TF.adjust_saturation(image, saturation_factor=random.uniform(0.7, 1.5))
        return image, mask, dist


    def __getitem__(self, idx):
        # Read RGB images
        img = io.imread(self.imgs[idx])# .astype(np.float32)
        # random.seed(seed)
        if self.img_transform:
            img_tensor = self.img_transform(img)
        else:
            img = img/np.max(img)
            img_tensor = torch.from_numpy(img).permute(2,0,1)

        # cell instance map 
        inst_map = np.load(self.anns[idx])# .astype("uint16")
        # create interior-edge map
        interior = np.zeros_like(inst_map, dtype=np.uint8)
        boundary = segmentation.find_boundaries(inst_map, mode='inner')
        boundary = morphology.binary_dilation(boundary, footprint=morphology.disk(1))
        interior_temp = np.logical_and(boundary == 0, inst_map > 0).astype(np.uint8)
        # interior_temp[boundary] = 0
        interior_temp = morphology.remove_small_objects(interior_temp==1)
        interior[interior_temp==1] = 1
        interior[boundary==1] = 2
        
        
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

        if self.mode == 'train':
            img_tensor, interior_tensor, inner_distance_tensor = self.my_transform(img_tensor, interior_tensor, inner_distance_tensor)  

        feed_dict = {"img": img_tensor, 'interior':interior_tensor, 'dist':inner_distance_tensor, 'name':self.imgs[idx]}


        return feed_dict




