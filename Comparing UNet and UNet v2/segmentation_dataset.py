"""This script defines a PyTorch Dataset class for loading and augmenting images and masks for lesion segmentation tasks.
It includes various data augmentation techniques such as flipping, rotation, brightness and contrast adjustments, Gaussian blur, color jittering, random cropping, and affine transformations.
The dataset is expected to be organized in two separate folders: one for images and one for masks. The images are resized to a specified size, and the masks are binarized.
The dataset can be used with PyTorch's DataLoader for training deep learning models.
"""
import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import random


class lesion_segmentation_dataset(Dataset):
    def __init__(self, images_folder, masks_folder, image_size=(256, 256), augment=True, normalize=True):
        self.images_folder = images_folder
        self.masks_folder = masks_folder
        self.image_size = image_size
        self.augment = augment
        self.normalize = normalize
        self.image_files = sorted(os.listdir(images_folder))
        self.mask_files = sorted(os.listdir(masks_folder))

        #using standard imagenet normalization values
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = os.path.join(self.images_folder, self.image_files[idx])
        mask_path = os.path.join(self.masks_folder, self.mask_files[idx])

        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L") 
        
        resize_needed = True #flag to check if resizing is needed
        
        #data augmentation
        if self.augment:
            if random.random() > 0.5:
                image = TF.hflip(image)
                mask = TF.hflip(mask)
            if random.random() > 0.5:
                image = TF.vflip(image)
                mask = TF.vflip(mask)
            if random.random() > 0.5:
                angle = random.uniform(-25, 25)
                image = TF.rotate(image, angle)
                mask = TF.rotate(mask, angle)
            if random.random() > 0.5:
                image = TF.adjust_brightness(image, brightness_factor=random.uniform(0.75, 1.25))
            if random.random() > 0.5:
                image = TF.adjust_contrast(image, contrast_factor=random.uniform(0.75, 1.25))
            if random.random() > 0.7:
                image = TF.gaussian_blur(image, kernel_size=5, sigma=random.uniform(0.3, 1.8))
            if random.random() > 0.3:
                color_jitter = transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.2, hue=0.05)
                image = color_jitter(image)
            if random.random() > 0.5:
                i, j, h, w = transforms.RandomResizedCrop.get_params(image, scale=(0.7, 1.0), ratio=(0.85, 1.15))
                image = TF.resized_crop(image, i, j, h, w, self.image_size)
                mask = TF.resized_crop(mask, i, j, h, w, self.image_size)
                resize_needed = False #no need to resize again'
            if random.random() > 0.4:
                params = transforms.RandomAffine.get_params(degrees=(-10, 10), translate=(0.05, 0.05), scale_ranges=None, shears=None, img_size=image.size)
                image = TF.affine(image, *params, interpolation=TF.InterpolationMode.BILINEAR)
                mask = TF.affine(mask, *params, interpolation=TF.InterpolationMode.NEAREST)
            if random.random() > 0.3:
                factor = random.uniform(0.9, 1.5)
                image = TF.adjust_sharpness(image, sharpness_factor=factor)
        
        #only resizing if not already resized
        if resize_needed:
            image = TF.resize(image, self.image_size)
            mask = TF.resize(mask, self.image_size)

        image = TF.to_tensor(image)
        mask = TF.to_tensor(mask)

        if self.normalize:
            image = TF.normalize(image, mean=self.mean, std=self.std)

        mask = (mask > 0.5).float() #binarizing mask

        return image, mask
