import os
import numpy as np
from skimage import io
from skimage.transform import resize
import torch
import torch.nn as nn
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import math
import json
from torchvision.utils import make_grid
from scipy.ndimage import convolve
import cv2

class FSIM:
    def __init__(self):
        # Parameters for the FSIM algorithm
        self.T1 = 0.85  # Constant for the phase congruency similarity
        self.T2 = 160   # Constant for the gradient magnitude similarity
        
        # Define the PC filters - 2D log-Gabor filters
        # These values are based on the original FSIM paper
        self.nscale = 4
        self.norient = 4
        self.minWaveLength = 6
        self.mult = 2
        self.sigmaOnf = 0.55
        self.k = 2.0
        self.cutOff = 0.5
        self.g = 10
        self.epsilon = 1e-10
        
    def compute_fsim(self, img1, img2):
        """
        Compute the Feature Similarity Index (FSIM) between two images.
        
        Parameters:
        -----------
        img1 : ndarray
            Reference image (grayscale, uint8)
        img2 : ndarray
            Distorted image (grayscale, uint8)
            
        Returns:
        --------
        fsim : float
            FSIM score (higher means more similar)
        """
        # Convert to float for computation
        img1 = img1.astype(np.float64)
        img2 = img2.astype(np.float64)
        
        # Step 1: Extract phase congruency (PC) maps
        PC1, G1 = self._compute_pc_and_gradient(img1)
        PC2, G2 = self._compute_pc_and_gradient(img2)
        
        # Step 2: Compute the phase congruency similarity (PC_m)
        PC_m = (2 * PC1 * PC2 + self.T1) / (PC1**2 + PC2**2 + self.T1)
        
        # Step 3: Compute the gradient magnitude similarity (G_m)
        G_m = (2 * G1 * G2 + self.T2) / (G1**2 + G2**2 + self.T2)
        
        # Step 4: Combine the two similarity measures
        S_L = G_m * PC_m
        
        # Step 5: Calculate the FSIM value
        PC_max = np.maximum(PC1, PC2)
        fsim = np.sum(S_L * PC_max) / np.sum(PC_max + self.epsilon)
        
        return fsim
    
    def _compute_pc_and_gradient(self, img):
        """
        Compute phase congruency (PC) and gradient magnitude for an image.
        
        This is a simplified implementation using the gradient magnitude
        and Laplacian of Gaussian for phase congruency approximation.
        """
        # Compute gradient magnitude
        gx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
        grad_mag = np.sqrt(gx**2 + gy**2)
        
        # Simplified phase congruency using Laplacian of Gaussian (LoG)
        # In real implementation, this would use log-Gabor filters
        log_kernel = np.array([
            [0, 1, 0],
            [1, -4, 1],
            [0, 1, 0]
        ], dtype=np.float64)
        
        pc = np.abs(convolve(img, log_kernel))
        
        # Normalize
        pc = pc / (np.max(pc) + self.epsilon)
        grad_mag = grad_mag / (np.max(grad_mag) + self.epsilon)
        
        return pc, grad_mag
    
    def compute_fsimc(self, img1, img2):
        """
        Compute the FSIM for color images (FSIMc).
        
        Parameters:
        -----------
        img1 : ndarray
            Reference color image (RGB, uint8)
        img2 : ndarray
            Distorted color image (RGB, uint8)
            
        Returns:
        --------
        fsimc : float
            FSIMc score (higher means more similar)
        """
        # Convert to float
        img1 = img1.astype(np.float64)
        img2 = img2.astype(np.float64)
        
        # Convert RGB to YIQ color space
        Y1, I1, Q1 = self._rgb_to_yiq(img1)
        Y2, I2, Q2 = self._rgb_to_yiq(img2)
        
        # Compute FSIM on luminance (Y) component
        PC1, G1 = self._compute_pc_and_gradient(Y1)
        PC2, G2 = self._compute_pc_and_gradient(Y2)
        
        # Phase congruency similarity
        PC_m = (2 * PC1 * PC2 + self.T1) / (PC1**2 + PC2**2 + self.T1)
        
        # Gradient magnitude similarity
        G_m = (2 * G1 * G2 + self.T2) / (G1**2 + G2**2 + self.T2)
        
        # Chrominance similarity
        I_m = (2 * I1 * I2 + self.T1) / (I1**2 + I2**2 + self.T1)
        Q_m = (2 * Q1 * Q2 + self.T1) / (Q1**2 + Q2**2 + self.T1)
        
        # Color similarity
        lambda_param = 0.03  # Weight parameter for chromatic components
        S_C = I_m * Q_m
        
        # Combined similarity
        S_L = PC_m * G_m
        S = S_L * (S_C ** lambda_param)
        
        # Calculate the FSIMc value
        PC_max = np.maximum(PC1, PC2)
        fsimc = np.sum(S * PC_max) / np.sum(PC_max + self.epsilon)
        
        return fsimc
    
    def _rgb_to_yiq(self, rgb_img):
        """Convert RGB image to YIQ color space."""
        if rgb_img.shape[2] != 3:
            raise ValueError("Input image must be RGB (3 channels)")
        
        # RGB to YIQ conversion matrix
        transform = np.array([
            [0.299, 0.587, 0.114],
            [0.596, -0.274, -0.322],
            [0.211, -0.523, 0.312]
        ])
        
        # Reshape to (n_pixels, 3)
        original_shape = rgb_img.shape
        img_reshaped = rgb_img.reshape(-1, 3)
        
        # Apply transformation
        yiq = np.dot(img_reshaped, transform.T)
        
        # Reshape back
        yiq_image = yiq.reshape(original_shape)
        
        # Return individual channels
        Y = yiq_image[:, :, 0]
        I = yiq_image[:, :, 1]
        Q = yiq_image[:, :, 2]
        
        return Y, I, Q


class Metrics():
    def __init__(self):
        import lpips
        self.lpips_model = lpips.LPIPS(net="alex").cuda()  # vgg squeeze
        self.lpips_model.eval()
        self.fsim = FSIM()  # Initialize FSIM
    
    def compute_psnr_batch(self, image_true, image_test):
        psnr_list = []
        if isinstance(image_true, torch.Tensor):
            image_true = image_true.to("cpu").numpy()
        if isinstance(image_test, torch.Tensor):
            image_test[image_test<0] = 0
            image_test[image_test>255] = 255
            image_test = image_test.to("cpu").numpy().astype('uint8')
            
        for i in range(image_true.shape[0]):
            target_image = image_true[i,:,:,:]
            reference_image = image_test[i,:,:,:]
            psnr_value = peak_signal_noise_ratio(reference_image, target_image, data_range=255.0)
            psnr_list.append(psnr_value)
        return sum(psnr_list)

    def compute_ssim_batch(self, image_true, image_test):
        ssim_list = []
        # Transform data type for index calculation
        if isinstance(image_true, torch.Tensor):
            image_true = image_true.to("cpu").numpy()
        if isinstance(image_test, torch.Tensor):
            image_test[image_test<0] = 0
            image_test[image_test>255] = 255
            image_test = image_test.to("cpu").detach().numpy().astype('uint8')
        for i in range(image_true.shape[0]):
            target_image = image_true[i,0,:,:]
            reference_image = image_test[i,0,:,:]
            ssim_value = structural_similarity(reference_image, target_image, win_size=7, data_range=255.0)
            ssim_list.append(ssim_value)
        return sum(ssim_list)
    
    def compute_lpips_batch(self, image_true, image_test):
        lpips_value = self.lpips_model(image_test, image_true)
        return lpips_value.sum()
    
    def compute_fsim_batch(self, image_true, image_test):
        """
        Compute FSIM for a batch of grayscale images
        """
        fsim_list = []
        # Transform data type
        if isinstance(image_true, torch.Tensor):
            image_true = image_true.to("cpu").numpy()
        if isinstance(image_test, torch.Tensor):
            image_test[image_test<0] = 0
            image_test[image_test>255] = 255
            image_test = image_test.to("cpu").detach().numpy().astype('uint8')
            
        for i in range(image_true.shape[0]):
            target_image = image_true[i,0,:,:]
            reference_image = image_test[i,0,:,:]
            fsim_value = self.fsim.compute_fsim(reference_image, target_image)
            fsim_list.append(fsim_value)
        return sum(fsim_list)
    
    def compute_fsimc_batch(self, image_true, image_test):
        """
        Compute FSIMc for a batch of color images
        """
        fsimc_list = []
        # Transform data type
        if isinstance(image_true, torch.Tensor):
            image_true = image_true.to("cpu").numpy()
        if isinstance(image_test, torch.Tensor):
            image_test[image_test<0] = 0
            image_test[image_test>255] = 255
            image_test = image_test.to("cpu").detach().numpy().astype('uint8')
            
        for i in range(image_true.shape[0]):
            # For color images (C, H, W) -> (H, W, C)
            target_image = np.transpose(image_true[i,:,:,:], (1, 2, 0))
            reference_image = np.transpose(image_test[i,:,:,:], (1, 2, 0))
            fsimc_value = self.fsim.compute_fsimc(reference_image, target_image)
            fsimc_list.append(fsimc_value)
        return sum(fsimc_list)
    
    def compute_resolution_batch(self, images, calibration_factor=3.3):
        """
        计算一批图像中的分辨率（两个可分辨点之间的最小距离）
        
        参数:
        images: 形状为(batch, 1, height, width)的图像批次
        calibration_factor: 从像素转换到实际单位的校准因子，这里是3.3微米/像素
        
        返回:
        resolution_values: 每个图像的分辨率值列表
        avg_resolution: 平均分辨率
        """
        import numpy as np
        from scipy import ndimage
        from skimage import feature
        import torch
        
        resolution_values = []
        
        # 转换为numpy数组
        if isinstance(images, torch.Tensor):
            images = images.to("cpu").detach().numpy()
            
        # 处理每个图像
        for i in range(images.shape[0]):
            # 获取单个图像并确保它是2D的（去除通道维度）
            image = images[i, 0, :, :]
            
            # 计算图像的梯度
            sx = ndimage.sobel(image, axis=0, mode='constant')
            sy = ndimage.sobel(image, axis=1, mode='constant')
            gradient = np.hypot(sx, sy).astype(np.float32)
            
            # 找到局部最大值（可能对应于点对）
            # 调整min_distance以匹配您的图像特征
            local_max = feature.peak_local_max(gradient, min_distance=5)
            
            # 计算所有点对之间的距离
            distances = []
            for j in range(len(local_max)):
                for k in range(j+1, len(local_max)):
                    point1 = local_max[j]
                    point2 = local_max[k]
                    distance = np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
                    distances.append(distance)
            
            # 找出最小距离（排除可能是噪音的非常小的值）
            if distances:
                valid_distances = [d for d in distances if d > 1.0]
                if valid_distances:
                    min_distance = min(valid_distances)
                    # 应用校准因子转换为微米
                    resolution = min_distance * calibration_factor
                    resolution_values.append(resolution)
                else:
                    # 如果没有有效距离，则返回None或默认值
                    resolution_values.append(None)
            else:
                resolution_values.append(None)  
        # 计算平均分辨率（排除None值）
        valid_resolutions = [r for r in resolution_values if r is not None]
        avg_resolution = sum(valid_resolutions) / len(valid_resolutions) if valid_resolutions else None
        
        return sum(resolution_values)# , avg_resolution
    
if __name__ =="__main__":
    metrics = Metrics()
    root_path = '/home/glm/EM_imaging_enhance/experiments/transfer_em_HeLa_without_finetune_250318_211510/results'
    imgs_path = os.listdir(root_path)
    imgs_path = [os.path.join(root_path, img_path) for img_path in imgs_path]
    imgs_path = sorted(imgs_path)
    psnr_sum = 0.0
    ssim_sum = 0.0
    lpips_sum = 0.0
    fsim_sum = 0.0
    resolution_GT_sum = 0.0
    resolution_SR_sum = 0.0
    
    for i in range(0,int(len(imgs_path)/3)):
        GT = torch.tensor(io.imread(imgs_path[i])).unsqueeze(0).unsqueeze(0)
        SR = torch.tensor(io.imread(imgs_path[i].replace('GT','Output'))).unsqueeze(0).unsqueeze(0)
        # Img = io.imread(imgs_path[i].replace('GT','Input'))
        psnr_sum += metrics.compute_psnr_batch(GT, SR)
        ssim_sum += metrics.compute_ssim_batch(GT, SR)
        lpips_sum += metrics.compute_lpips_batch(GT.cuda(), SR.cuda())
        fsim_sum += metrics.compute_fsim_batch(GT, SR)
        resolution_GT_sum += metrics.compute_resolution_batch(GT)
        resolution_SR_sum += metrics.compute_resolution_batch(SR)
    print('psnr:',psnr_sum/len(imgs_path)*3)
    print('ssim:',ssim_sum/len(imgs_path)*3)
    print('lpips:',lpips_sum/len(imgs_path)*3)
    print('fsim:',fsim_sum/len(imgs_path)*3)
    print('resolution_GT:',resolution_GT_sum/len(imgs_path)*3)
    print('resolution_SR:',resolution_SR_sum/len(imgs_path)*3)
    print('resolution ratio:',resolution_SR_sum/resolution_GT_sum)


    
