# Computer-Vision-Projects
A couple of deep learning models for training neural networks for medical image analysis using PyTorch

# Folder Descriptions

 - Comparing U-Net and U-Net v2 
    * A comparative research project evaluating U-Net and U-Net v2 architectures for skin lesion segmentation on the ISIC 2017 dataset. This study investigates how architectural differences, pretraining, data augmentation, and hyperparameter tuning affect segmentation  
      performance.
      
    * File description:
      - segmentation_dataset.py: PyTorch dataset class for loading and augmenting images and segmentation masks from dataset.
      - unet_model.py: Baseline U-Net implementation using a ResNet encoder backbone with support for feature map extraction.
      - UNet_v2.py: U-Net v2 implementation including the SDI (Semantic and Detail Infusion) module and PVTv2 transformer-based encoder.
      - unet_main.py: Training and evaluation script for vanilla U-Net; includes model logging, metric tracking, and visualization.
      - unetv2_main.py: Training and evaluation script for U-Net v2 with support for pretraining, test-time augmentation (TTA), and deep supervision.
      - pvtv2.py: Definition of Pyramid Vision Transformer v2 (PVTv2) variants (B0–B5) used as the encoder for U-Net v2.
      - paper/Comparing U-Net and U-Net v2 for ISIC 2017 Skin Lesion Segmentation: This paper discusses architectural differences, performance comparisons, effects of data augmentation, deep supervision, thresholding, and pretraining, supported with metrics, ablation 
        studies, and training curves.
        
    * Dataset Source: https://challenge.isic-archive.com/data/#2017
      
    * Citation for Dataset: 
      Codella N, Gutman D, Celebi ME, Helba B, Marchetti MA, Dusza S, Kalloo A, Liopyris K, Mishra N, Kittler H, Halpern A. "Skin Lesion Analysis Toward Melanoma Detection: A Challenge at the 2017 International Symposium on Biomedical Imaging (ISBI), Hosted by the   
      International Skin Imaging Collaboration (ISIC)". arXiv: 1710.05006 [cs.CV]
     
      
 - ResNet152 Classification with Heatmaps
    * A classification project using a pretrained ResNet152 model on the ISIC 2016 Task 3 dataset for melanoma detection. Grad-CAM is used to visualize class-discriminative regions through activation heatmaps at various ResNet layers.
      
    * File description:
      - dataset.py: Custom dataset loader with support for labels and preprocessing. Reading image dataset and groundtruth .csv file
      - model.py: ResNet model altered with wrapper to extract intermediate feature map from each layer for visualization
      - grad_cam.py: for hooking target layer and computing gradients to output heatmaps
      - main.py: Training and evaluation script with metrics computation (accuracy, AUC, sensitivity, specificity), ROC curve plotting, and Weights & Biases (wandb) integration.
        
    * Dataset Source: https://challenge.isic-archive.com/data/#2016
      
    * Citation for Dataset: 
      Gutman, David; Codella, Noel C. F.; Celebi, Emre; Helba, Brian; Marchetti, Michael; Mishra, Nabin; Halpern, Allan. "Skin Lesion Analysis toward Melanoma Detection: A Challenge at the International Symposium on Biomedical Imaging (ISBI) 2016, hosted by the         
      International Skin Imaging Collaboration (ISIC)". eprint arXiv:1605.01397. 2016.


