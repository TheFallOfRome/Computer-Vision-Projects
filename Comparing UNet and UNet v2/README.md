## Comparing U-Net and U-Net v2 for ISIC 2017 Skin Lesion Segmentation

This project is a comparative study of the **U-Net** and **U-Net v2** deep learning architectures on the ISIC 2017 skin lesion segmentation task. It investigates the impact of architectural differences, data augmentation, loss function design, and pretraining on segmentation performance.

**Paper:** [Comparing U-Net and U-Net v2 for ISIC 2017 Skin Lesion Segmentation](https://github.com/TheFallOfRome/Computer-Vision-Projects/blob/150b4d5687b2952fcbe68fedb4af0840c6c729b0/Comparing%20UNet%20and%20UNet%20v2/paper/Comparing%20U-Net%20and%20U-Net%20v2%20for%20ISIC%202017%20Skin%20Lesion%20Segmentation.pdf)  
**Dataset Source:** [ISIC 2017 Challenge](https://challenge.isic-archive.com/data/#2017)

### Note
  - U-Net v2 was implemented initially without pretraining by default for research purposes, but pretraining with PVTv2_b2 found online or using 'timm' can be enabled in 'unetv2_main.py'
  - Vanilla U-Net comes in 'unet_model.py' and was used without pretraining initally be default as well. Pretraining can be enabled by installing 'segmentation_models_pytorch' and initializing it in 'unet_main.py'.
  - The ISIC 2017 Dataset comes with superpixel masks in .png format, and must be removed from the Training, Validation, and Test data files found in the link provided. The Groundtruth files come set for inference. 

### Visualization and Metrics
  - Prediction masks, training & validation loss curves, and final metric scores (Dice and IoU) are automatically saved during evaluation
    * Visual Triplet Output is saved in a user specified path as:
      - [Test Image | Predicted Mask | Ground Truth Mask]
    * Metrics Tracked
      - Dice Coefficient, Intersection over Union, Training & Validation Loss
        
