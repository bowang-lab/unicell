#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 26 17:02:47 2022

@author: jma
"""
import argparse
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from dataloaders.pngtif_loader import PNGTIFLoader
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from datetime import datetime
import os
join = os.path.join
import monai
from tqdm import tqdm
from monai.transforms import Compose, EnsureType, Activations, AsDiscrete
import shutil
from models.unicell_modules import UniCell
from utils.multi_task_sliding_window_inference import multi_task_sliding_window_inference
from optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
torch.autograd.set_detect_anomaly(True)



print('Successfully import all requirements!')

parser = argparse.ArgumentParser('UniCell for Microscopy image segmentation', add_help=False)
# Dataset parameters
parser.add_argument('--data_path', default='/media/jma/sg/junma/UniCell/AllCellImgs/', type=str,
                    help='training data path; subfolders: images, labels')
parser.add_argument('--dataset_name', default='cellImgAll', type=str, help='training data name; used in model path')
parser.add_argument('--work_dir', default='./work_dir',
                    help='path where to save models and logs')
parser.add_argument('--seed', default=42, type=int)
parser.add_argument('--resume', default=False, help='resume from checkpoint')
parser.add_argument('--num_workers', default=6, type=int)

# Model parameters
parser.add_argument('--model_folder_name', default='unicell', type=str, help='folder name to save models')
parser.add_argument('--num_class', default=3, type=int, help='segmentation classes')
parser.add_argument('--input_size', default=256, type=int, help='segmentation classes')

# Training parameters
parser.add_argument('--pre_train', default=False, help='use pretrained model')
parser.add_argument('--pre_train_path', default='/home/jma/Documents/CellSegBaseline/work_dir/b2unetcellseg1k/', help='use pretrained model')
parser.add_argument('--batch_size', default=32, type=int, help='Batch size per GPU')
parser.add_argument('--max_epochs', default=4000, type=int)
parser.add_argument('--val_interval', default=5, type=int) 
parser.add_argument('--epoch_tolerance', default=100, type=int)
parser.add_argument('--initial_lr', type=float, default=6e-4, help='learning rate')

# def main():
#     args = parser.parse_args()
args = parser.parse_args()

#%% set training/validation split
np.random.seed(args.seed)
img_path = join(args.data_path, 'images')
gt_path = join(args.data_path, 'labels')

img_names = sorted(os.listdir(img_path))
gt_names = [img_name.split('.')[0]+'_label.tiff' for img_name in img_names]
img_num = len(img_names)
val_frac = 0.0
indices = np.arange(img_num)
np.random.shuffle(indices)
val_split = int(img_num*val_frac)
train_indices = indices[val_split:]
val_indices = indices[:val_split]

images_Tr = [join(img_path, img_names[i]) for i in train_indices]
labels_Tr = [join(gt_path, gt_names[i]) for i in train_indices]
images_Val = [join(img_path, img_names[i]) for i in val_indices]
labels_Val = [join(gt_path, gt_names[i]) for i in val_indices]

data_Tr = PNGTIFLoader(images_Tr, labels_Tr, crop_size=args.input_size, mode='train')
train_loader = DataLoader(data_Tr, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=True)
data_Val = PNGTIFLoader(images_Tr[::8], labels_Tr[::8], mode='validation')
val_loader = DataLoader(data_Val, num_workers=1, batch_size=1, shuffle=False)
print(f"training image num: {len(data_Tr)}, validation image num: {len(val_loader)}")

#%% set parameters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('training is on', device)
max_epochs = args.max_epochs
num_class = 3 # background, interior, edge
torch.backends.cudnn.benchmark = True
  
work_dir = './work_dir'
model_path = join(args.work_dir, args.model_folder_name)
os.makedirs(model_path, exist_ok=True)
run_id = datetime.now().strftime("%Y%m%d-%H%M")
shutil.copyfile(__file__, join(model_path, run_id + '_' + os.path.basename(__file__)))


model = UniCell(in_channels=3, out_channels=3, regress_class=1, img_size=args.input_size)
model.to(device)

if args.pre_train:
    model.load_state_dict(torch.load(join(args.pre_train_path, 'latest_model.pth'), map_location=device))
    print('loaded pretrined model:', join(args.pre_train_path, 'latest_model.pth'))
loss_interoir = monai.losses.DiceCELoss(to_onehot_y=False, softmax=False)
loss_dist = nn.MSELoss()
optimizer = torch.optim.AdamW(model.parameters(), args.initial_lr, weight_decay=1e-5)
scheduler = LinearWarmupCosineAnnealingLR(optimizer, warmup_epochs=50, max_epochs=max_epochs)
# optimizer = torch.optim.SGD(model.parameters(), initial_lr, decay,momentum=0.99, nesterov=True)
epoch_loss_values = list()
# set metric track
val_interval = args.val_interval
dice_metric = monai.metrics.DiceMetric(include_background=False, reduction="mean", get_not_nans=False)
mse_metric = monai.metrics.MSEMetric(reduction="mean", get_not_nans=False)
post_pred = Compose([EnsureType(), Activations(softmax=False), AsDiscrete(threshold=0.5)])
post_gt = Compose([EnsureType(), AsDiscrete(to_onehot=None)])
best_dice = 0
best_dice_epoch = 0
best_F1 = 0
best_F1_epoch = 0
epoch_tolerance = 300
val_dice_values = []
val_mse_values = []
val_f1_values = []
val_dice_values2 = []
print(f'{model_path} Training start....')
writer = SummaryWriter(model_path)
#%% training
for epoch in range(1, max_epochs+1):
    model.train()
    epoch_loss = 0
    for step, batch_data in enumerate(tqdm(train_loader)):
        pixel_values = batch_data['img'].type(torch.FloatTensor).to(device) # (b, 3, 256,256)
        labels = batch_data['interior'].long().to(device) # (b,1,256,256)
        gt_interior = monai.networks.one_hot(labels, num_class) # (b,cls,256,256)
        gt_dist = batch_data['dist'].type(torch.FloatTensor).cuda() # (b, 1, 256,256) 
        
        optimizer.zero_grad()
        logits_full, pred_dist_full = model(pixel_values)
        # resample to original resolution
        # logits_full = nn.functional.interpolate(logits_low, size=labels.shape[-2:], mode="bilinear", align_corners=False)
        pred_interior = torch.softmax(logits_full, dim=1)
        # pred_dist = nn.functional.interpolate(dists_low, size=labels.shape[-2:], mode="bilinear", align_corners=False)     
        pred_dist = torch.sigmoid(pred_dist_full) # normalize network outputs to [0,1]
        # conpute loss
        loss_eval_interior = loss_interoir(pred_interior, gt_interior)
        loss_eval_dist = loss_dist(pred_dist, gt_dist)
        # interior loss + distance loss
        loss_final = loss_eval_interior # + loss_eval_dist
        loss_final.backward()
        optimizer.step()
        epoch_loss+=loss_final.item()
    epoch_loss /= step
    epoch_loss_values.append(epoch_loss)
    scheduler.step()
    writer.add_scalar("train_loss", epoch_loss, epoch)
    checkpoint = {'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': epoch_loss_values,
            }
    print('epoch:{}/{}, loss:{:.4f}, DiceCE loss:{:.4f}, MSE:{:.4f}'.format(epoch, max_epochs, epoch_loss, loss_eval_interior.item(), loss_eval_dist.item()))
    # validation
    # if epoch>20 and epoch % val_interval == 0:
    #     model.eval()
    #     with torch.no_grad():
    #         val_images = None
    #         val_labels = None
    #         val_seg = None
    #         val_seg_insts = []
    #         val_gt_insts = []
    #         for val_data in val_loader:
    #             val_images = val_data['img'].type(torch.FloatTensor).to(device) # (b, 3, 256,256)
    #             # val_gt_dist = val_data['dist'].type(torch.FloatTensor).to(device) # (b, 1, 256,256) 
    #             labels_ori = val_data['interior'].long().to(device) # (b,1,256,256)
    #             val_gt_inst = val_data['inst'].numpy() # (b, 256, 256), 'int16'
                
    #             val_labels = monai.networks.one_hot(labels_ori, num_class)
    #             # get model outputs
    #             val_pred, val_pred_dist = multi_task_sliding_window_inference(inputs=val_images, roi_size=(args.input_size, args.input_size), sw_batch_size=4, predictor=model)
    #             # val_pred = nn.functional.interpolate(val_logits_low, size=val_labels.shape[-2:], mode="bilinear", align_corners=False)
    #             # val_pred = torch.softmax(upsampled_logits,1)
    #             # val_pred_dist = nn.functional.interpolate(val_dists_low, size=val_labels.shape[-2:], mode="bilinear", align_corners=False)    
    #             # val_pred_dist = torch.sigmoid(val_pred_dist)


    #             val_seg = [post_pred(i) for i in monai.data.decollate_batch(val_pred[:,0:2,:,:])]
    #             val_gt = [post_gt(i) for i in monai.data.decollate_batch(val_labels[:,0:2,:,:])]
    #             print(os.path.basename(val_data['name'][0]), dice_metric(y_pred=val_seg, y=val_gt))
    #             # watershed postprocessing
    #             # val_seg_inst = watershed_post(val_pred_dist.squeeze(1).cpu().numpy(), val_pred.squeeze(1).cpu().numpy()[:,1])
    #             # val_seg_insts.append(val_seg_inst.squeeze()) 
    #             # val_gt_insts.append(val_gt_inst.squeeze())
    #         # aggregate the final mean dice
    #         metric = dice_metric.aggregate().item()
    #         val_dice_values.append(metric)

    #         # compute object-level metric
    #         # val_seg_insts = np.concatenate(val_seg_insts, axis=0)
    #         # val_gt_insts = np.concatenate(val_gt_insts, axis=0)
    #         # val_inst_metric = eval_img_list(val_gt_insts, val_seg_insts)
    #         # val_F1 = np.mean(val_inst_metric['f1'])
    #         # val_f1_values.append(val_F1)
    #         # val_dice2 = np.mean(val_inst_metric['dice'])
    #         # val_dice_values2.append(val_dice2)            
    #         # print('epoch:{}, validation Dice:{:.4f}, F1:{:.4f}, inst Dice:{:.4f}'.format(epoch, metric, val_F1, val_dice2))
    #         print('epoch:{}, validation Dice:{:.4f}'.format(epoch, metric))
    #         # reset the status for next validation round
    #         dice_metric.reset()
            
    #         if metric > best_dice:
    #             best_dice = metric
    #             best_dice_epoch = epoch
    #             torch.save(checkpoint, join(model_path, 'best_dice.pth'))
    #         else:
    #             print(f'No improvement in validation metric! best_Dice_epoch:{best_dice_epoch}, best_Dice:{best_dice}')
    #         writer.add_scalar("val_mean_dice", metric, epoch)

            # if val_F1>best_F1:
            #     best_F1 = val_F1
            #     best_F1_epoch = epoch
            #     torch.save(model.state_dict(), join(model_path, 'best_F1.pth'))
            # else:
            #     print(f'No improvement in validation F1 metric! best_F1_epoch:{best_F1_epoch}, best_F1:{best_F1}')

#     if (epoch-best_dice_epoch)>epoch_tolerance and (epoch-best_F1_epoch)>epoch_tolerance:
#         print(f"validation loss does not improve for {epoch_tolerance} epoches! Current epoch:{epoch}, best Dice epoch:{best_dice_epoch}, best Dice:{best_dice}")
#         torch.save(model.state_dict(), join(model_path, 'epoch_'+str(epoch)+'.pth'))
#         break   
writer.close()
# print('model save path:{}, best_dice_epoch:{}, best_dice:{}, current epoch:{}'.format(model_path, best_dice_epoch, best_dice, epoch))
# np.savez_compressed(join(model_path, 'train_log.npz'), val_dice=val_dice_values, val_f1=val_f1_values, inst_dice=val_dice_values2, epoch_loss=epoch_loss_values)


torch.save(checkpoint, join(model_path, 'final_model.pth'))
np.savez_compressed(join(model_path, 'train_log.npz'), val_dice=val_dice_values, epoch_loss=epoch_loss_values)














