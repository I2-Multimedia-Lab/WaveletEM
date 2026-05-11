import os
import torch
import model as Model
import argparse
import logging
import core.logger as Logger
from dataset import MyDataset
from torch.utils.data import DataLoader
from model.reconstruction_network import CDFormer_SR
from tqdm import tqdm
from PIL import Image
import pywt
import numpy as np
import torch.nn as nn
import torch.optim as optim
from core.metrics import Metrics
from timm.models.layers import DropPath, trunc_normal_

def wavelet_reconstruct(input_tensor):
       
        assert input_tensor.dim() == 4, 
        assert input_tensor.size(1) == 4, 
        batch_size, _, half_size, _ = input_tensor.size()
        image_size = half_size * 2
     
        output_tensor = torch.zeros(
            batch_size, 1, image_size, image_size, 
            dtype=input_tensor.dtype, 
            device=input_tensor.device
        )
     
        for b in range(batch_size):
           
            cA = input_tensor[b, 0].cpu().detach().numpy()
            cH = input_tensor[b, 1].cpu().detach().numpy()
            cV = input_tensor[b, 2].cpu().detach().numpy()
            cD = input_tensor[b, 3].cpu().detach().numpy()
          
            reconstructed = pywt.idwt2((cA, (cH, cV, cD)), 'haar')
          
            output_tensor[b, 0] = torch.from_numpy(reconstructed).cuda()
        return output_tensor

file_name = 'sr_em_250226_162430'
root_path = os.path.join('./experiments',file_name)

parser = argparse.ArgumentParser()
mode = 'denoise' if file_name.startswith('denoise') else 'sr'
if mode == 'denoise':
    parser.add_argument('-c', '--config', type=str, default='config/Denoise_256.jsonc',
                        help='JSON file for configuration')
    test_dataset = MyDataset(data_json_path='./emdiffuse_dataset/denoise_test_256/test_image_path.json',mode='test')
    # train_dataset = MyDataset(data_json_path='./emdiffuse_dataset/denoise_train_256/train_image_path.json')
else:
    parser.add_argument('-c', '--config', type=str, default='config/SR_128_256.jsonc',
                        help='JSON file for configuration')
    test_dataset = MyDataset(data_json_path='./emdiffuse_dataset/super_resolution_test_128/test_image_path.json',mode='test')
    # train_dataset = MyDataset(data_json_path='./emdiffuse_dataset/super_resolution_train_128/train_image_path.json')
parser.add_argument('-p', '--phase', type=str, choices=['train', 'val'],
                    help='Run either train(training) or val(generation)', default='train')
parser.add_argument('-gpu', '--gpu_ids', type=str, default=None)
parser.add_argument('-debug', '-d', action='store_true')
parser.add_argument('-enable_wandb', action='store_true')
parser.add_argument('-log_wandb_ckpt', action='store_true')
parser.add_argument('-log_eval', action='store_true')

test_dataloader = DataLoader(dataset=test_dataset,batch_size=64,num_workers=4)

# parse configs
args = parser.parse_args()
opt = Logger.parse(args)
# Convert to NoneDict, which return None for missing key.
opt = Logger.dict_to_nonedict(opt)
opt['phase'] = 'val'

diffusion = Model.create_model(opt)


diffusion.set_new_noise_schedule(opt['model']['beta_schedule']['val'], schedule_phase='val')

state_dict = torch.load(os.path.join(root_path, 'checkpoint', 'current_gen.pth'))
diffusion.netG.load_state_dict(state_dict, strict=False)

transformer = CDFormer_SR().cuda()
transformer.load_state_dict(torch.load(os.path.join(root_path,'checkpoint','transformer_gen.pth')))
transformer.eval()
metrics = Metrics()

j = 0
avg_psnr = 0
avg_ssim = 0
avg_lpips = 0
avg_fsim = 0
for _,val_data in enumerate(tqdm(test_dataloader,desc="Validation")):
    with torch.no_grad():
        data = {"HR":val_data['gt_lf'],"SR":val_data['img_lf']}
        diffusion.feed_data(data)
        diffusion.test(continous=False)
        visuals = diffusion.get_current_visuals()
        sr_img_hf = transformer(val_data['img_hf'].cuda())*255.0
        sr_img_lf = (visuals['SR']+1)/2 * 510.0
        sr_wt = torch.cat([sr_img_lf.cuda(),sr_img_hf],dim=1)
        sr = wavelet_reconstruct(sr_wt).cpu()
        # sr = model(sr_wt).cpu()
        # sr = (sr+1)/2*255.0
        val_data['gt'] = (val_data['gt']+1)/2*255.0
        val_data['img'] = (val_data['img']+1)/2*255.0

        for i in range(sr.size(0)):
            out = sr[i].squeeze().numpy()
            out = out.astype('uint8')
            out = Image.fromarray(out, 'L')
            out.save(os.path.join(opt['path']['results'],f'Output_{j}.tif'))

            input = val_data['img'][i].squeeze().numpy()
            input = input.astype('uint8')
            input = Image.fromarray(input, 'L')
            input.save(os.path.join(opt['path']['results'],f'Input_{j}.tif'))

            gt = val_data['gt'][i].squeeze().numpy()
            gt = gt.astype('uint8')
            gt = Image.fromarray(gt, 'L')
            gt.save(os.path.join(opt['path']['results'],f'GT_{j}.tif'))
            j += 1
        avg_psnr += metrics.compute_psnr_batch(val_data['gt'],sr)
        avg_ssim += metrics.compute_ssim_batch(val_data['gt'],sr)
        avg_lpips+=metrics.compute_lpips_batch((val_data['gt']).cuda(),sr.cuda())
        # avg_fsim+=metrics.compute_fsim_batch((val_data['gt']).cuda(),sr.cuda())
        avg_fsim+=metrics.compute_resolution_batch(sr.cuda())
psnr = avg_psnr / test_dataset.__len__()
ssim = avg_ssim / test_dataset.__len__()
lpips = avg_lpips / test_dataset.__len__()
fsim = avg_fsim / test_dataset.__len__()
print(f"Validation PSNR:{psnr};SSIM:{ssim};LPIPS:{lpips};FSIM:{fsim}")
print(f"Validation PSNR:{psnr};SSIM:{ssim};LPIPS:{lpips}")
