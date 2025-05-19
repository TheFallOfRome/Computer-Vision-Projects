import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
import skimage
from skimage.io import imread
from skimage.transform import resize
import random
import numpy as np


class SkinLesionDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = self.data.iloc[idx, 0]
        #print('img_name', img_name)
        img_path = self.root_dir + '/' + img_name + '.jpg'
        image = Image.open(img_path).convert('RGB')
        label = self.data.iloc[idx, 1]
        #print('label', label)
        if self.transform:
            image = self.transform(image)

        return image, label





class MySkinLesionDataset(Dataset):
    def __init__(self, csv_file, root_dir2):
        self.data = pd.read_csv(csv_file)
        self.root_dir2 = root_dir2
        self.label_map = { 'benign': 0, 'malignant': 1}

    def __len__(self):
        total_num = len(self.data)
        #print('number of data', total_num)
        return total_num

    def __getitem__(self, idx):
        img_name = self.data.iloc[idx, 0]
        #print('img_name', img_name)
       


        img_path = self.root_dir2 + '/' + img_name + '.jpg'
        


        #image = Image.open(img_path).convert('RGB')
        big_tI = imread(img_path)
        #print('big_tI shape', big_tI.shape)
        target_size = (224, 224)
        big_tI_resized = resize(big_tI, target_size, preserve_range=True, anti_aliasing=True)
        big_tI2 = skimage.img_as_float(big_tI_resized)




        



        
        nrotate = random.randint(0, 3)
        #train_sample = np.rot90(big_tI, nrotate)
        train_sample2 = np.rot90(big_tI2, nrotate)
        nflip = random.randint(0, 1)
        if nflip:
            #train_sample = np.fliplr(train_sample)
            train_sample2 = np.fliplr(train_sample2)
            

        #print('train_sample min', train_sample.min())
        #print('train_sample max', train_sample.max())
        ### do a normalization 
        

        # mean=[0.485, 0.456, 0.406]
        # std=[0.229, 0.224, 0.225]


        # for i in range(train_sample.shape[2]):  # Loop over channels
        #     if i % 3 == 0:
        #         train_sample[:, :, i] = (train_sample[:, :, i] - mean[0]) / std[0]
        #     elif i % 3 == 1:
        #         train_sample[:, :, i] = (train_sample[:, :, i] - mean[1]) / std[1]
        #     elif i % 3 == 2:
        #         train_sample[:, :, i] = (train_sample[:, :, i] - mean[2]) / std[2]
               
                


        #print('train_sample after min', train_sample.min())
        #print('train_sample after max', train_sample.max())


        # train_sample = np.transpose(train_sample, (2, 0, 1))
        # #print('train_sample  shape', train_sample.shape)
        # #print('train_sample  ', (train_sample))
        # train_sample = torch.from_numpy(np.array(train_sample))
        #print('train_sample  ', (train_sample))




        
        #print('big_tI shape', big_tI.shape)
        
        #big_tI = imread(img_path)
        #big_tI = skimage.img_as_float(big_tI)
        

        #print('big_tI', big_tI)
        #print('big_tI min', big_tI.min())
        #print('big_tI max', big_tI.max())
        #big_tI = resize(big_tI, (224, 224))
        #print('big_tI after resize shape', big_tI.shape)
        #print('big_tI', big_tI)
        #print('big_tI min', big_tI.min())
        #print('big_tI max', big_tI.max())

        # apply extra augmentation (same random operation on both image and label)
        
        
            

        #print('train_sample min', train_sample.min())
        #print('train_sample max', train_sample.max())
        ### do a normalization 
        

        mean2=[0.485, 0.456, 0.406]
        std2=[0.229, 0.224, 0.225]


        for i in range(train_sample2.shape[2]):  # Loop over channels
            train_sample2[:, :, i] = (train_sample2[:, :, i] - mean2[i]) / std2[i]
         
               
                


        #print('train_sample after min', train_sample.min())
        #print('train_sample after max', train_sample.max())


        train_sample2 = np.transpose(train_sample2, (2, 0, 1))
        #print('train_sample  shape', train_sample.shape)
        #print('train_sample  ', (train_sample))
        train_sample2 = torch.from_numpy(np.array(train_sample2))


        #print('train_sample  shape', train_sample.shape)
        #print('train_sample2  shape', train_sample2.shape)
        train_sample_final = train_sample2 # torch.cat((train_sample, train_sample2), 0)
        #print('train_sample_final  shape', train_sample_final.shape)

        
        raw_label = self.data.iloc[idx, 1]
        if isinstance(raw_label, str):
            label_tensor = torch.tensor(self.label_map[raw_label])
        else:
            label_tensor = torch.tensor(int(raw_label))
        #print('label_tensor', label_tensor)
        #print('pd shape', pd.shape)
        return train_sample_final, label_tensor








