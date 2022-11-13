#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 31 18:10:52 2022

@author: 12593
"""

import numpy as np
from numba import jit
from scipy.optimize import linear_sum_assignment
from collections import OrderedDict
import pandas as pd
from tqdm import tqdm

def mask_ious(masks_true, masks_pred):
    """ return best-matched masks """
    iou = _intersection_over_union(masks_true, masks_pred)[1:,1:]
    n_min = min(iou.shape[0], iou.shape[1])
    costs = -(iou >= 0.5).astype(float) - iou / (2*n_min)
    true_ind, pred_ind = linear_sum_assignment(costs)
    iout = np.zeros(masks_true.max())
    iout[true_ind] = iou[true_ind,pred_ind]
    preds = np.zeros(masks_true.max(), 'int')
    preds[true_ind] = pred_ind+1
    return iout, preds

def _intersection_over_union(masks_true, masks_pred):
    """ intersection over union of all mask pairs
    
    Parameters
    ------------
    
    masks_true: ND-array, int 
        ground truth masks, where 0=NO masks; 1,2... are mask labels
    masks_pred: ND-array, int
        predicted masks, where 0=NO masks; 1,2... are mask labels

    Returns
    ------------

    iou: ND-array, float
        matrix of IOU pairs of size [x.max()+1, y.max()+1]
    
    ------------
    How it works:
        The overlap matrix is a lookup table of the area of intersection
        between each set of labels (true and predicted). The true labels
        are taken to be along axis 0, and the predicted labels are taken 
        to be along axis 1. The sum of the overlaps along axis 0 is thus
        an array giving the total overlap of the true labels with each of
        the predicted labels, and likewise the sum over axis 1 is the
        total overlap of the predicted labels with each of the true labels.
        Because the label 0 (background) is included, this sum is guaranteed
        to reconstruct the total area of each label. Adding this row and
        column vectors gives a 2D array with the areas of every label pair
        added together. This is equivalent to the union of the label areas
        except for the duplicated overlap area, so the overlap matrix is
        subtracted to find the union matrix. 

    """
    overlap = _label_overlap(masks_true, masks_pred)
    n_pixels_pred = np.sum(overlap, axis=0, keepdims=True)
    n_pixels_true = np.sum(overlap, axis=1, keepdims=True)
    iou = overlap / (n_pixels_pred + n_pixels_true - overlap)
    iou[np.isnan(iou)] = 0.0
    return iou

def _true_positive(iou, th):
    """ true positive at threshold th
    
    Parameters
    ------------

    iou: float, ND-array
        array of IOU pairs
    th: float
        threshold on IOU for positive label

    Returns
    ------------

    tp: float
        number of true positives at threshold
        
    ------------
    How it works:
        (1) Find minimum number of masks
        (2) Define cost matrix; for a given threshold, each element is negative
            the higher the IoU is (perfect IoU is 1, worst is 0). The second term
            gets more negative with higher IoU, but less negative with greater
            n_min (but that's a constant...)
        (3) Solve the linear sum assignment problem. The costs array defines the cost
            of matching a true label with a predicted label, so the problem is to 
            find the set of pairings that minimizes this cost. The scipy.optimize
            function gives the ordered lists of corresponding true and predicted labels. 
        (4) Extract the IoUs fro these parings and then threshold to get a boolean array
            whose sum is the number of true positives that is returned. 

    """
    n_min = min(iou.shape[0], iou.shape[1])
    costs = -(iou >= th).astype(float) - iou / (2*n_min)
    true_ind, pred_ind = linear_sum_assignment(costs)
    match_ok = iou[true_ind, pred_ind] >= th
    tp = match_ok.sum()
    return tp


def average_precision(masks_true, masks_pred, threshold=[0.5, 0.75, 0.9]):
    """ average precision estimation: AP = TP / (TP + FP + FN)

    This function is based heavily on the *fast* stardist matching functions
    (https://github.com/mpicbg-csbd/stardist/blob/master/stardist/matching.py)

    Parameters
    ------------
    
    masks_true: list of ND-arrays (int) or ND-array (int) 
        where 0=NO masks; 1,2... are mask labels
    masks_pred: list of ND-arrays (int) or ND-array (int) 
        ND-array (int) where 0=NO masks; 1,2... are mask labels

    Returns
    ------------

    ap: array [len(masks_true) x len(threshold)]
        average precision at thresholds
    tp: array [len(masks_true) x len(threshold)]
        number of true positives at thresholds
    fp: array [len(masks_true) x len(threshold)]
        number of false positives at thresholds
    fn: array [len(masks_true) x len(threshold)]
        number of false negatives at thresholds

    """
    not_list = False
    if not isinstance(masks_true, list):
        masks_true = [masks_true]
        masks_pred = [masks_pred]
        not_list = True
    if not isinstance(threshold, list) and not isinstance(threshold, np.ndarray):
        threshold = [threshold]
    ap  = np.zeros((len(masks_true), len(threshold)), np.float32)
    tp  = np.zeros((len(masks_true), len(threshold)), np.float32)
    fp  = np.zeros((len(masks_true), len(threshold)), np.float32)
    fn  = np.zeros((len(masks_true), len(threshold)), np.float32)
    n_true = np.array(list(map(np.max, masks_true)))
    n_pred = np.array(list(map(np.max, masks_pred)))
#     if len(n_pred) < 1:
#         n_pred = [0]
    for n in range(len(masks_true)):
        #_,mt = np.reshape(np.unique(masks_true[n], return_index=True), masks_pred[n].shape)
        if n_pred[n] > 0:
            iou = _intersection_over_union(masks_true[n], masks_pred[n])[1:, 1:]
            for k,th in enumerate(threshold):
                tp[n,k] = _true_positive(iou, th)
        fp[n] = n_pred[n] - tp[n]
        fn[n] = n_true[n] - tp[n]
        ap[n] = tp[n] / (tp[n] + fp[n] + fn[n]) # this is the jaccard index, not precision, right? 
        
    if not_list:
        ap, tp, fp, fn = ap[0], tp[0], fp[0], fn[0]
    return ap, tp, fp, fn

@jit(nopython=True)
def _label_overlap(x, y):
    """ fast function to get pixel overlaps between masks in x and y 
    
    Parameters
    ------------

    x: ND-array, int
        where 0=NO masks; 1,2... are mask labels
    y: ND-array, int
        where 0=NO masks; 1,2... are mask labels

    Returns
    ------------

    overlap: ND-array, int
        matrix of pixel overlaps of size [x.max()+1, y.max()+1]
    
    """
    x = x.ravel()
    y = y.ravel()
    
    # preallocate a 'contact map' matrix
    overlap = np.zeros((1+x.max(),1+y.max()), dtype=np.uint)
    
    # loop over the labels in x and add to the corresponding
    # overlap entry. If label A in x and label B in y share P
    # pixels, then the resulting overlap is P
    # len(x)=len(y), the number of pixels in the whole image 
    for i in range(len(x)):
        overlap[x[i],y[i]] += 1
    return overlap

def jaccard(gt, seg):
    if np.count_nonzero(gt)==0 and np.count_nonzero(seg)==0:
        jaccard_score = 1.0
    elif np.count_nonzero(gt)==0 and np.count_nonzero(seg)>0:
        jaccard_score = 0.0
    else:
        union = np.count_nonzero(np.logical_and(gt, seg))
        intersection = np.count_nonzero(np.logical_or(gt, seg))
        jaccard_score = union/intersection
    return jaccard_score

def dice(gt, seg):
    if np.count_nonzero(gt)==0 and np.count_nonzero(seg)==0:
        dice_score = 1.0
    elif np.count_nonzero(gt)==0 and np.count_nonzero(seg)>0:
        dice_score = 0.0
    else:
        union = np.count_nonzero(np.logical_and(gt, seg))
        intersection = np.count_nonzero(gt) + np.count_nonzero(seg)
        dice_score = 2*union/intersection
    return dice_score

def eval_tp_fp_fn(masks_true, masks_pred, threshold=0.5):
    num_inst_gt = np.max(masks_true)
    num_inst_seg = np.max(masks_pred)
    if num_inst_seg>0:
        iou = _intersection_over_union(masks_true, masks_pred)[1:, 1:]
            # for k,th in enumerate(threshold):
        tp = _true_positive(iou, threshold)
        fp = num_inst_seg - tp
        fn = num_inst_gt - tp
    else:
        print('No segmentation results!')
        tp = 0
        fp = 0
        fn = 0
        
    return tp, fp, fn

def eval_img_batch(y_gt, y_pred, threshold=0.5, res_format='DataFrame'):
    """
    Parameters
    ----------
    y_gt : np.array,  (B, H, W)
        batch of ground truth.
    y_pred : np.array (B, H, W)
        batch of instance segmentation results.

    Returns
    -------
    metrics dataframe.

    """
    assert y_gt.shape==y_pred.shape, print('Shape error between GT and segmentation!')
    num = y_pred.shape[0]
    metrics = OrderedDict()
    metrics['precision'] = []
    metrics['recall'] = []
    metrics['f1'] = []
    metrics['jaccard'] = []
    metrics['dice'] = []
    for i in tqdm(range(num)):#y_gt.shape[0]
        gt_i, seg_i = y_gt[i], y_pred[i]
        gt_i, _, _ = segmentation.relabel_sequential(gt_i)
        tp_i, fp_i, fn_i = eval_tp_fp_fn(gt_i, seg_i, threshold)
        if tp_i == 0:
            precision_i = 0
            recall_i = 0
            f1_i = 0
            print('ID:', i, 'no segmentation resultss!')
        else:
            precision_i = tp_i/(tp_i+fp_i)
            recall_i = tp_i/(tp_i+fn_i)
            f1_i = 2*(precision_i*recall_i)/(precision_i+recall_i)
        jaccard_i = jaccard(gt_i>0, seg_i>0)
        dice_i = dice(gt_i>0, seg_i>0)
        
        metrics['precision'].append(precision_i)
        metrics['recall'].append(recall_i)
        metrics['f1'].append(f1_i)
        metrics['jaccard'].append(jaccard_i)
        metrics['dice'].append(dice_i)
    if res_format=='DataFrame':
        return pd.DataFrame(metrics)
    elif res_format=='F1':
        return np.mean(metrics['f1'])
    else:
        return np.mean(metrics['jaccard'])

def eval_img_list(y_gts, y_preds, threshold=0.5, res_format='DataFrame'):
    """
    Parameters
    ----------
    y_gt : list  (B, H, W)
        batch of ground truth.
    y_pred : list (B, H, W)
        batch of instance segmentation results.

    Returns
    -------
    metrics dataframe.

    """
    assert len(y_gts)==len(y_preds), print('Shape error between GT and segmentation!')
    # num = y_pred.shape[0]
    metrics = OrderedDict()
    metrics['precision'] = []
    metrics['recall'] = []
    metrics['f1'] = []
    metrics['jaccard'] = []
    metrics['dice'] = []
    for i, gt_i, seg_i in enumerate(zip(y_gts, y_preds)):
        # gt_i, seg_i = y_gt[i], y_pred[i]
        tp_i, fp_i, fn_i = eval_tp_fp_fn(np.squeeze(gt_i), np.squeeze(seg_i), threshold)
        if tp_i == 0:
            precision_i = 0
            recall_i = 0
            f1_i = 0
            print('ID:', i, 'no segmentation resultss!')
        else:
            precision_i = tp_i/(tp_i+fp_i)
            recall_i = tp_i/(tp_i+fn_i)
            f1_i = 2*(precision_i*recall_i)/(precision_i+recall_i)
        jaccard_i = jaccard(gt_i>0, seg_i>0)
        dice_i = dice(gt_i>0, seg_i>0)
        
        metrics['precision'].append(precision_i)
        metrics['recall'].append(recall_i)
        metrics['f1'].append(f1_i)
        metrics['jaccard'].append(jaccard_i)
        metrics['dice'].append(dice_i)
    if res_format=='DataFrame':
        return pd.DataFrame(metrics)
    elif res_format=='F1':
        return np.mean(metrics['f1'])
    else:
        return np.mean(metrics['jaccard'])







