import os
join = os.path.join
import argparse
import numpy as np
import torch
import torch.nn as nn
import tifffile as tif
import monai
from tqdm import tqdm
from utils.postprocess import mask_overlay
from monai.transforms import Activations, AddChanneld, AsChannelFirstd, AsDiscrete, Compose, EnsureTyped, EnsureType
import matplotlib.pyplot as plt
from skimage import io, exposure, segmentation, morphology
from utils.postprocess import watershed_post
from models.unicell_modules import UniCell
from utils.multi_task_sliding_window_inference import multi_task_sliding_window_inference

def normalize_channel(img, lower=0.1, upper=99.9):
    non_zero_vals = img[np.nonzero(img)]
    percentiles = np.percentile(non_zero_vals, [lower, upper])
    if percentiles[1] - percentiles[0] > 0.001:
        img_norm = exposure.rescale_intensity(img, in_range=(percentiles[0], percentiles[1]), out_range='uint8')
    else:
        img_norm = img
    return img_norm

def main():
    parser = argparse.ArgumentParser('Baseline for Microscopy image segmentation', add_help=False)
    # Dataset parameters
    parser.add_argument('-i', '--input_path', default='/media/jma/sg/junma/UniCell/AllCellImgs/CellUniverse/imagesTs', type=str,
                        help='testing data path')
    parser.add_argument('-o','--out_path', default='./seg-imagesTs', type=str, help='output path')
    parser.add_argument('-m', '--model_path', default='./work_dir/demo', help='path where to save models and segmentation results')
    parser.add_argument('-pre', '--pretrain_model', default='unicell', help='pretrained model: unicell or uninuclei; if use customized model, please set None and specify model path')
    parser.add_argument('--contour_overlay', required=False, default=False, action="store_true", help='save segmentation boundary overlay')
    parser.add_argument('--overlay_color', required=False, default='green', type=str, help='color of overlay contour: white, green, blue, yellow, black')
    parser.add_argument('--mask_overlay', required=False, default=False, action="store_true", help='save segmentation mask overlay')
    parser.add_argument('--overwrite', default=False, action='store_true', required=False, help='overwrite existing segmentation results')
    # Model parameters
    parser.add_argument('--num_class', default=3, type=int, help='segmentation classes')
    parser.add_argument('--input_size', default=256, type=int, help='segmentation classes')
    args = parser.parse_args()

    os.makedirs(args.out_path, exist_ok=True)
    all_names = sorted(os.listdir(join(args.input_path)))
    if args.overwrite:
        img_names = all_names
    else:
        img_names = [i for i in all_names if not os.path.exists(join(args.out_path, i.split('.')[0]+'_label.tiff'))]


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = UniCell(in_channels=3, out_channels=args.num_class, regress_class=1, img_size=args.input_size).to(device)

    if args.pretrain_model == 'unicell':
        if os.path.isfile(join(os.path.dirname(__file__), 'work_dir/unicell/model.pth')):
            checkpoint = torch.load(join(os.path.dirname(__file__), 'work_dir/unicell/model.pth'), map_location=torch.device(device))
        else:
            os.makedirs(join(os.path.dirname(__file__), 'work_dir/unicell'), exist_ok=True)
            torch.hub.download_url_to_file('https://zenodo.org/record/7308987/files/model.pth?download=1', join(os.path.dirname(__file__), 'work_dir/unicell/model.pth'))
            checkpoint = torch.load(join(os.path.dirname(__file__), 'work_dir/unicell/model.pth'), map_location=torch.device(device))
    elif args.pretrain_model == 'uninuclei':
        if os.path.isfile(join(args.work_dir, 'uninuclei/model.pth')):
            checkpoint = torch.load(join(args.work_dir, 'uninuclei/model.pth'), map_location=torch.device(device))
        else:
            os.makedirs(join(args.work_dir, 'unicell'), exist_ok=True)
            torch.hub.download_url_to_file('https://zenodo.org/record/7308990/files/model.pth?download=1', join(args.work_dir, 'uninuclei/model.pth'))
            checkpoint = torch.load(join(args.work_dir, 'uninuclei/model.pth'), map_location=torch.device(device))
    else:
        checkpoint = torch.load(join(args.model_path, 'model.pth'))


    model.load_state_dict(checkpoint['model_state_dict'])
    post_pred = Compose([EnsureType(), Activations(softmax=False), AsDiscrete(threshold=0.5)])
    #%%
    roi_size = (args.input_size, args.input_size)
    sw_batch_size = 8
    model.eval()
    with torch.no_grad():
        for name in tqdm(img_names):
            if name.endswith('.tif') or name.endswith('.tiff'):
                img_data = tif.imread(join(args.input_path, name))
            else:
                img_data = io.imread(join(args.input_path, name))
            if len(img_data.shape)==2:
                pre_img_data = normalize_channel(img_data, lower=0.1, upper=99.9)
                pre_img_data = np.repeat(np.expand_dims(pre_img_data, -1), repeats=3, axis=-1)
            else:
                pre_img_data = np.zeros((img_data.shape[0], img_data.shape[1], 3), dtype=np.uint8)
                for i in range(3):
                    img_channel_i = img_data[:,:,i]
                    if len(img_channel_i[np.nonzero(img_channel_i)])>0:
                        pre_img_data[:,:,i] = normalize_channel(img_channel_i, lower=0.1, upper=99.9)
            test_npy = pre_img_data/np.max(pre_img_data)
            test_tensor = torch.from_numpy(np.expand_dims(test_npy, 0)).permute(0,3,1,2).type(torch.FloatTensor).to(device)
            
            val_pred, val_pred_dist = multi_task_sliding_window_inference(inputs=test_tensor, roi_size=(args.input_size, args.input_size), sw_batch_size=4, predictor=model)

            val_seg = [post_pred(i) for i in monai.data.decollate_batch(val_pred[:,0:2,:,:])]

            # watershed postprocessing
            val_seg_inst = watershed_post(val_pred_dist.squeeze(1).cpu().numpy(), val_pred.squeeze(1).cpu().numpy()[:,1])        
            test_pred_mask = val_seg_inst.squeeze().astype(np.uint16)
            tif.imwrite(join(args.out_path, name.split('.')[0]+'_label.tiff'), test_pred_mask, compression='zlib')

            if args.contour_overlay:
                boundary = segmentation.find_boundaries(test_pred_mask, connectivity=1, mode='inner')
                boundary = morphology.binary_dilation(boundary, morphology.disk(1))
                if args.overlay_color == 'white':
                    pre_img_data[boundary, :] = 255
                elif args.overlay_color == 'green':
                    pre_img_data[boundary, 1:2] = 255 # green channel
                    pre_img_data[boundary, 0:1] = 0
                    pre_img_data[boundary, 2:3] = 0
                elif args.overlay_color == 'yellow':
                    pre_img_data[boundary, 1:2] = 255 # green channel
                    pre_img_data[boundary, 0:1] = 255 # red channel
                    pre_img_data[boundary, 2:3] = 0   
                elif args.overlay_color == 'blue':
                    pre_img_data[boundary, 1:2] = 0
                    pre_img_data[boundary, 0:1] = 0
                    pre_img_data[boundary, 2:3] = 255 # blue channel                

                io.imsave(join(args.out_path, 'overlay_contour_' + name.split('.')[0]+'.png'), pre_img_data, check_contrast=False)
            if args.mask_overlay:
                img_mask_overlay = mask_overlay(pre_img_data, test_pred_mask)
                io.imsave(join(args.out_path, 'overlay_mask_' + name.split('.')[0]+'.png'), img_mask_overlay, check_contrast=False)

if __name__ == '__main__':
    main()


    






