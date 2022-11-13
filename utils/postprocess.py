#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  7 10:51:48 2022

@author: jma
"""

import numpy as np
from skimage import segmentation, measure, exposure, morphology
import scipy.ndimage as nd
from tqdm import tqdm
import skimage 
import colorsys

def fill_holes(label_img, size=10, connectivity=1):
    output_image = np.copy(label_img)
    props = measure.regionprops(np.squeeze(label_img.astype('int')), cache=False)
    for prop in props:
        if prop.euler_number < 1:

            patch = output_image[prop.slice]

            filled = morphology.remove_small_holes(
                ar=(patch == prop.label),
                area_threshold=size,
                connectivity=connectivity)

            output_image[prop.slice] = np.where(filled, prop.label, patch)

    return output_image

def watershed_post(distmaps, interiors, interior_thre=0.2, dist_thre=0.1):
    """
    Parameters
    ----------
    distmaps : float (N, H, W) N is the number of cells
        distance transform map of cell/nuclear [0,1].
    interiors : float (N, H, W)
        interior map of cell/nuclear [0,1].

    Returns
    -------
    label_images : uint (N, H, W)
        cell/nuclear instance segmentation.

    """
    
    label_images = []
    for distmap, interior in zip(distmaps, interiors):# in interiors[0:num]:
        interior = nd.gaussian_filter(interior.astype(np.float32), 2)
        # find marker based on distance map
        if skimage.__version__ > '0.18.2':
            markers = measure.label(morphology.h_maxima(image=distmap, h=dist_thre, footprint=morphology.disk(2)))
        else:
            markers = measure.label(morphology.h_maxima(image=distmap, h=dist_thre, selem=morphology.disk(2)))
        # print('distmap marker num:', np.max(markers), 'interior marker num:', np.max(makers_interior))
        
        label_image = segmentation.watershed(-1 * interior, markers,
                                mask=interior > interior_thre, 
                                watershed_line=0)

        label_image = morphology.remove_small_objects(label_image, min_size=15)
        # fill in holes that lie completely within a segmentation label
        label_image = fill_holes(label_image, size=15)

        # Relabel the label image
        label_image, _, _ = segmentation.relabel_sequential(label_image)
        label_images.append(label_image)
    label_images = np.stack(label_images, axis=0).astype(np.uint)
    return label_images



def hsv_to_rgb(arr):
    hsv_to_rgb_channels = np.vectorize(colorsys.hsv_to_rgb)
    h, s, v = np.rollaxis(arr, axis=-1)
    r, g, b = hsv_to_rgb_channels(h, s, v)
    rgb = np.stack((r,g,b), axis=-1)
    return rgb

def mask_overlay(img, masks):
    """ overlay masks on image (set image to grayscale)
    Adapted from https://github.com/MouseLand/cellpose/blob/06df602fbe074be02db3d716e280f0990816c726/cellpose/plot.py#L172
    Parameters
    ----------------

    img: int or float, 2D or 3D array
        img is of size [Ly x Lx (x nchan)]

    masks: int, 2D array
        masks where 0=NO masks; 1,2,...=mask labels

    Returns
    ----------------

    RGB: uint8, 3D array
        array of masks overlaid on grayscale image

    """

    if img.ndim>2:
        img = img.astype(np.float32).mean(axis=-1)
    else:
        img = img.astype(np.float32)
    
    HSV = np.zeros((img.shape[0], img.shape[1], 3), np.float32)
    HSV[:,:,2] = np.clip((img / 255. if img.max() > 1 else img) * 1.5, 0, 1)
    hues = np.linspace(0, 1, masks.max()+1)[np.random.permutation(masks.max())]
    for n in range(int(masks.max())):
        ipix = (masks==n+1).nonzero()
        HSV[ipix[0],ipix[1],0] = hues[n]

        HSV[ipix[0],ipix[1],1] = 1.0
    RGB = (hsv_to_rgb(HSV) * 255).astype(np.uint8)
    return RGB









