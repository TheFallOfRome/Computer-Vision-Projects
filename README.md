# Computer-Vision-Projects
A couple of deep learning models for training neural networks for medical image analysis using PyTorch

# Folder Descriptions
 - ResNet152 Classification with Heatmaps
    * A classication task using pretrained ResNet152 orginally on the ISIC 2016 Task 3 dermoscopic lesion dataset. Includes extraction and visualization of feature maps using Grad-CAM by generating heatmaps per activation map layer.
      
    * File description:
      - dataset.py: for reading image dataset and grountruth .csv file
      - model.py: ResNet model altered with wrapper to output feature map from each layer for visualization
      - grad_cam.py: for hooking target layer and computing gradients to output heatmaps
      - main.py: implementing model and training on dataset. Includes "wandb" integration for veiwing model training data. Additionally includes computing of model performance, sensitivity and specificity, and plottinf ROC curve.
        
    * Link to originally used dataset: https://challenge.isic-archive.com/data/#2016
      
    * Citation for the dataset: 
      Gutman, David; Codella, Noel C. F.; Celebi, Emre; Helba, Brian; Marchetti, Michael; Mishra, Nabin; Halpern, Allan. "Skin Lesion Analysis toward Melanoma Detection: A Challenge at the International Symposium on Biomedical Imaging (ISBI) 2016, hosted by the         
      International Skin Imaging Collaboration (ISIC)". eprint arXiv:1605.01397. 2016.


