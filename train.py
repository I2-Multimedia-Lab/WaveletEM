import torch
import model as Model
import argparse
import logging
import core.logger as Logger
from core.metrics import Metrics
from core.wandb_logger import WandbLogger
import os
import numpy as np
from dataset import MyDataset
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
tqdm(ascii=True)
from model.reconstruction_network import CDFormer_SR
import pywt
from PIL import Image

class Trainer:
    def __init__(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('-c', '--config', type=str, default='config/Denoise_256.jsonc',
                            help='JSON file for configuration')
        parser.add_argument('-p', '--phase', type=str, choices=['train', 'val'],
                            help='Run either train(training) or val(generation)', default='train')
        parser.add_argument('-gpu', '--gpu_ids', type=str, default=None)
        parser.add_argument('-debug', '-d', action='store_true')
        parser.add_argument('-enable_wandb', action='store_true')
        parser.add_argument('-log_wandb_ckpt', action='store_true')
        parser.add_argument('-log_eval', action='store_true')

        # parse configs
        args = parser.parse_args()
        opt = Logger.parse(args)
        # Convert to NoneDict, which return None for missing key.
        self.opt = Logger.dict_to_nonedict(opt)
        #log
        Logger.setup_logger(logger_name='train_loger',root=self.opt['path']['log'],phase='train', level=logging.INFO, screen=False)
        self.logger = logging.getLogger("train_loger")
        self.logger.info(Logger.dict2str(self.opt))
        #dataset
        train_dataset = MyDataset(data_json_path=self.opt['datasets']['train']['dataroot'],mode=self.opt['phase'])
        self.train_dataloader = DataLoader(train_dataset,batch_size=self.opt['datasets']['train']['batch_size'],
                                           shuffle=self.opt['datasets']['train']['use_shuffle'],num_workers=self.opt['datasets']['train']['num_workers'])
        self.test_dataset = MyDataset(data_json_path=self.opt['datasets']['val']['dataroot'],mode='test')
        #model
        self.diffusion = Model.create_model(self.opt)
        self.metrics = Metrics()
        self.transformer = CDFormer_SR().cuda()
        # self.transformer.load_state_dict(torch.load(os.path.join('./experiments/sr_em_250213_004046','checkpoint','transformer_gen.pth')))
        # self.forzen_checkpoint()
        # # train
        self.optim = torch.optim.Adam(self.transformer.parameters(),lr=1e-4,betas=(0.9,0.99))
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optim,step_size=20,gamma=0.5)
        self.loss = torch.nn.L1Loss()
    def forzen_checkpoint(self):
        for param in self.transformer.parameters():
            param.requires_grad = False
    def wavelet_reconstruct(self,input_tensor):

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

    def train_one_epoch(self,epoch,total_epoch):
        
        diffusion_loss = 0.0
        transformer_loss = 0.0
        
        self.diffusion.set_new_noise_schedule(
            self.opt['model']['beta_schedule'][self.opt['phase']], schedule_phase=self.opt['phase'])
        self.transformer.train()
        for _,train_data in enumerate(tqdm(self.train_dataloader,desc=f'Training {epoch+1} / {total_epoch} epochs')):
            data = {"HR":train_data['gt_lf'],"SR":train_data['img_lf']}
            self.diffusion.feed_data(data)
            self.diffusion.optimize_parameters()
            logs = self.diffusion.get_current_log()
            for k,v in logs.items():diffusion_loss += v

            self.optim.zero_grad()
            lr_hf = train_data['img_hf'].cuda()
            hr_hf = train_data['gt_hf'].cuda()
            sr_hf = self.transformer(lr_hf)
            loss = self.loss(sr_hf,hr_hf)
            loss.backward()
            self.optim.step()
            transformer_loss += loss.item()
        self.logger.info(f"Training {epoch+1} / {total_epoch} epochs  Diffusion loss:{diffusion_loss}")
        self.logger.info(f"Training {epoch+1} / {total_epoch} epochs  Transformer loss:{transformer_loss}")

        return diffusion_loss ,transformer_loss

    def validation(self):
        avg_psnr = 0.0
        avg_ssim = 0.0
        avg_lpips = 0.0
       
        n = self.opt["datasets"]["val"]["data_len"]
        indices = torch.randperm(len(self.test_dataset)).tolist()
        subset_indices = indices[:n]
        subset = Subset(self.test_dataset, subset_indices)
        test_dataloader = DataLoader(subset, batch_size=10, shuffle=True)
        
        self.transformer.eval()
        self.diffusion.set_new_noise_schedule(
            self.opt['model']['beta_schedule']['val'], schedule_phase='val')
        
        with torch.no_grad(): 
            for _,  val_data in enumerate(tqdm(test_dataloader,desc="Validation")):
                data = {"HR":val_data['gt_lf'],"SR":val_data['img_lf']}
                self.diffusion.netG.denoise_fn.cdp = None
                self.diffusion.feed_data(data)
                self.diffusion.test(continous=False)
                visuals = self.diffusion.get_current_visuals()
                # sr_img = self.metrics.tensor2img(visuals['SR'])  # uint8
                # hr_img = self.metrics.tensor2img(visuals['HR']) # uint8
                
                sr_img_hf = self.transformer(val_data['img_hf'].cuda())*255.0
                sr_img_lf = (visuals['SR']+1)/2 * 510.0
                sr_wt = torch.cat([sr_img_lf.cuda(),sr_img_hf],dim=1)
                sr = self.wavelet_reconstruct(sr_wt).cpu()
                # sr = ((visuals['SR']+1)/2*255.0).cpu()
                val_data['gt'] = (val_data['gt']+1)/2*255.0

                avg_psnr += self.metrics.compute_psnr_batch(val_data['gt'],sr)
                avg_ssim += self.metrics.compute_ssim_batch(val_data['gt'],sr)
                avg_lpips+=self.metrics.compute_lpips_batch((val_data['gt']).cuda(),sr.cuda())

            psnr = avg_psnr / n
            ssim = avg_ssim / n
            lpips = avg_lpips / n
            self.logger.info(f"Validation PSNR:{psnr};SSIM:{ssim};LPIPS:{lpips}")
            return psnr,ssim,lpips

    def train(self,total_epoch):
        psnr_best = 0.0
        ssim_best = 0.0
        for i in range(0,total_epoch):
            loss = self.train_one_epoch(epoch=i,total_epoch=total_epoch)
            self.diffusion.schedulerG.step()
            self.scheduler.step()
            print(f"training loss:{loss}")
            psnr,ssim,lpips = self.validation()
            print(f"PSNR:{psnr};SSIM:{ssim};LPIPS:{lpips}")
            if psnr >= psnr_best:
                psnr_best = psnr
                self.save_model()
            elif ssim >= ssim_best:
                ssim_best = ssim
                self.save_model()
            else:
                pass
    
    def test(self):
        self.test_dataloader = DataLoader(self.test_dataset,batch_size=32,shuffle=False)
        self.transformer.eval()
        self.diffusion.set_new_noise_schedule(self.opt['model']['beta_schedule']['val'], schedule_phase='val')
        j = 0
        avg_psnr = 0.0
        avg_ssim = 0.0
        avg_lpips = 0.0
        with torch.no_grad():
            for _, val_data in enumerate(tqdm(self.test_dataloader,desc="Test")):
                data = {"HR":val_data['gt_lf'],"SR":val_data['img_lf']}
                self.diffusion.netG.denoise_fn.cdp = None
                self.diffusion.feed_data(data)
                self.diffusion.test(continous=False)
                visuals = self.diffusion.get_current_visuals()
                sr_img_hf = self.transformer(val_data['img_hf'].cuda())*255.0
                sr_img_lf = (visuals['SR']+1)/2 * 510.0
                sr_wt = torch.cat([sr_img_lf.cuda(),sr_img_hf],dim=1)
                sr = self.wavelet_reconstruct(sr_wt).cpu()
                # sr = ((visuals['SR']+1)/2*255.0).cpu()
                val_data['gt'] = (val_data['gt']+1)/2*255.0
                val_data['img'] = (val_data['img']+1)/2*255.0

                for i in range(sr.size(0)):
                    out = sr[i].squeeze().numpy()
                    out = out.astype('uint8')
                    out = Image.fromarray(out, 'L')
                    out.save(os.path.join(self.opt['path']['results'],f'Output_{j}.tif'))

                    input = val_data['img'][i].squeeze().numpy()
                    input = input.astype('uint8')
                    input = Image.fromarray(input, 'L')
                    input.save(os.path.join(self.opt['path']['results'],f'Input_{j}.tif'))

                    gt = val_data['gt'][i].squeeze().numpy()
                    gt = gt.astype('uint8')
                    gt = Image.fromarray(gt, 'L')
                    gt.save(os.path.join(self.opt['path']['results'],f'GT_{j}.tif'))
                    j += 1
                avg_psnr += self.metrics.compute_psnr_batch(val_data['gt'],sr)
                avg_ssim += self.metrics.compute_ssim_batch(val_data['gt'],sr)
                avg_lpips+=self.metrics.compute_lpips_batch((val_data['gt']).cuda(),sr.cuda())
        psnr = avg_psnr / self.test_dataset.__len__()
        ssim = avg_ssim / self.test_dataset.__len__()
        lpips = avg_lpips / self.test_dataset.__len__()
        self.logger.info(f"Test PSNR:{psnr};SSIM:{ssim};LPIPS:{lpips}")        

    def save_model(self):
        self.diffusion.save_network()
        self.logger.info("Diffusion Model saved")

        save_path = os.path.join(self.opt['path']['checkpoint'], 'transformer_gen.pth')
        state_dict = self.transformer.state_dict()
        for key, param in state_dict.items():
            state_dict[key] = param.cpu()
        # 保存权重
        torch.save(state_dict, save_path)
        self.logger.info("Transformer Model saved")
        self.test()


if __name__ == "__main__":
    train = Trainer()
    # train.validation()
    train.train(120)
    # train.save_model()
