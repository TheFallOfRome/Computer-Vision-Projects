import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, confusion_matrix, auc, roc_curve
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, classification_report
from model import ResNet152
import matplotlib.pyplot as plt
import wandb
from argparse import ArgumentParser
import os
import numpy as np
import random
from classification_dataset import Skin_Lesion_Dataset
import time
import math
from grad_cam import GradCAM
import cv2

#os.environ["CUDA_VISIBLE_DEVICES"] = "1"



def main(args):
    if config.wandb:
          print(f"[wandb] Tracking via wandb - Name: {config.run_name}")
          wandb.init(
              id=wandb.util.generate_id(),
              project='',
              name=config.run_name, 
              config=config,
              settings=wandb.Settings(_executable='<executable>'),
          )

    # Setup seed and cuda setting
    seed = 1
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True  # fixed input/model: ~5-10% speedup

    
    model = ResNet152(num_classes=config.num_classes)
    # Move the model to the desired device (CPU or GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    tot_parameters = sum([p.numel() for p in model.parameters()])
    print(f'💠 Model initiated with {tot_parameters} parameters')



    train_dataset = Skin_Lesion_Dataset(config.train_csv,  config.training_dataset)
    test_dataset = Skin_Lesion_Dataset(config.test_csv,  config.testing_dataset)
    print('total trainig samples', len(train_dataset))
    print('total testing samples', len(test_dataset))
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size)

    #using focal loss due to class imbalance
    class FocalLoss(nn.Module):
        def __init__(self, alpha=1, gamma=2, reduction='mean'):
            super(FocalLoss, self).__init__()
            self.alpha = alpha
            self.gamma = gamma
            self.reduction = reduction

        def forward(self, inputs, targets):
            ce_loss = nn.CrossEntropyLoss(reduction='none')(inputs, targets)
            pt = torch.exp(-ce_loss)
            focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

            if self.reduction == 'mean':
                return focal_loss.mean()
            elif self.reduction == 'sum':
                return focal_loss.sum()
            else:
                return focal_loss
  
    criterion = FocalLoss(alpha=0.75, gamma=2)
    #criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=0.005, betas=(0.9, 0.999))

    if not os.path.exists(config.model_path):
        os.makedirs(config.model_path)
   

    # Training loop
    for epoch in range(1, config.num_epochs):
        
        model.train()
        running_loss = 0.0
        time1 = time.time()
        for images, labels in train_loader:
            images = images.float().to(device)
            labels = labels.to(device)

     


            # Forward pass
            outputs, _ = model(images)


            loss_1 = criterion(outputs, labels)

        

            loss = loss_1 

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if config.wandb:
              wandb.log({'train_loss': running_loss})
        
        time2 = time.time()
        print('-----------------------------------')
        print('Epoch {}, total time {:.2f}'.format(epoch, time2 - time1))
        print('train_loss', running_loss)

        # update learning rate
        cur_lr = config.lr * math.pow(1 - epoch/config.num_epochs, 0.9)
        for param_group in optimizer.param_groups:
            param_group['lr'] = cur_lr
        print('Updated lr:', cur_lr)

        if epoch%config.saved_epochs ==0 or epoch==2:
          # checkpoint_name = os.path.join(config.model_path, f'{config.run_name}_epoch{epoch}.pth')
          # torch.save(model.state_dict(),os.path.join(config.model_path, checkpoint_name))
          
          model.eval()
          test_predictions = []
          test_labels = []
          test_predicted_probs = []
          
          #making directories for heatmaps
          os.makedirs("gradcam_outputs/heatmaps", exist_ok=True)
          os.makedirs("gradcam_outputs/overlays", exist_ok=True)
          os.makedirs("gradcam_outputs/originals", exist_ok=True)
          
          #parameters for conditional loop during heatmap making
          max_gradcam_images = 4
          gradcam_image_count = 0
          
          #layers for heatmap generation
          target_layers = [model.layer1[-1], model.layer2[-1], model.layer3[-1], model.layer4[-1]]
          layer_names = ["layer1", "layer2", "layer3", "layer4"]

          #testing
          for images, labels in test_loader:
              images = images.float().to(device)
              labels = labels.to(device)
              
              #test loop predictions and labels
              with torch.no_grad():
                  outputs, _ = model(images)

                  #print('outputs', outputs)
                  prob = torch.softmax(outputs, dim=-1)
                  
                  test_predicted_probs.extend(prob.cpu().numpy())
                  predict = torch.argmax(prob, dim=1)

                  #print('predict', predict)
                  test_predictions.extend(predict.cpu().numpy())
                  test_labels.extend(labels.cpu().numpy())

              if gradcam_image_count < max_gradcam_images:

                for i in range(images.size(0)):
                    
                    if gradcam_image_count >= max_gradcam_images:
                        break
                    
                    #getting images for grad cam
                    image_single = images[i].unsqueeze(0)
                    pred_class = predict[i].item()

                    #enabling gradients 
                    image_single.requires_grad = True

                    #forward pass
                    model.zero_grad()
                    output, _ = model(image_single)

                    class_score = output[0, pred_class]
                    class_score.backward()
                    
                    #getting gradients
                    img_np = image_single.squeeze().permute(1, 2, 0).cpu().detach().numpy()
                    img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())
                    img_np = np.uint8(255 * img_np)
                    
                    #saving original image
                    cv2.imwrite(f'gradcam_outputs/originals/original_{gradcam_image_count}.png', img_np)

                    if config.wandb:
                        wandb.log({
                            f'Original Image {gradcam_image_count + 1}': wandb.Image(f'gradcam_outputs/originals/original_{gradcam_image_count}.png')
                        })

                    for layer, name in zip(target_layers, layer_names):
                        #generating heatmap
                        gradcam = GradCAM(model, layer)
                        heatmap = gradcam.generate(image_single, target_class=pred_class)
                        
                        #using Dr. Gu's exmaple (Lecture slides 20)
                        heatmap_resized = cv2.resize(heatmap, (img_np.shape[1], img_np.shape[0]))
                        heatmap_uint8 = np.uint8(255 * heatmap_resized)
                        heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

                        superimposed_img = cv2.addWeighted(img_np, 0.6, heatmap_colored, 0.4, 0)
                        
                        #saving heatmaps and overlays in cluster and wandb
                        heatmap_path = f'gradcam_outputs/heatmaps/heatmap_img{gradcam_image_count + 1}_{name}.png'
                        overlay_path = f'gradcam_outputs/overlays/overlay_img{gradcam_image_count + 1}_{name}.png'

                        cv2.imwrite(heatmap_path, heatmap_colored)
                        cv2.imwrite(overlay_path, superimposed_img)

                        if config.wandb:
                            wandb.log({
                                f'Heatmap {gradcam_image_count + 1} - {name}': wandb.Image(heatmap_path),
                                f'Overlay {gradcam_image_count + 1} - {name}': wandb.Image(overlay_path)
                            })

                    gradcam_image_count += 1



                  
          

          test_predicted_probs = np.vstack(test_predicted_probs)
          # print('test_predicted_probs', test_predicted_probs)
          # print('test_labels', test_labels)
          # print('test_predictions', test_predictions)


          test_accuracy = accuracy_score(test_labels, test_predictions)
          test_accuracy = "{:.4f}".format(test_accuracy)
          print('test_accuracy ', test_accuracy)

          f1 = f1_score(test_labels, test_predictions)
          print(f"F1 Score: {f1:.4f}")


          test_auc_ovr = roc_auc_score(test_labels, test_predicted_probs[:,1]) # for binary
          #test_auc_ovr = roc_auc_score(test_labels, test_predicted_probs, multi_class="ovr")
          print('test_auc_ovr ', test_auc_ovr)


          cm = confusion_matrix(test_labels, test_predictions)  
          #print('cm', cm)
          
          sensitivity = np.mean(sen(config.num_classes, cm))
          sensitivity = "{:.4f}".format(sensitivity)
          print('sensitivity ', sensitivity)


          specificity = np.mean(spe(config.num_classes, cm))
          specificity = "{:.4f}".format(specificity)
          print('specificity ', specificity)

          print(classification_report(test_labels, test_predictions, target_names=["benign", "malignant"])) 
         
          fpr, tpr, thresholds = roc_curve(test_labels, test_predicted_probs[:, 1])  # Assumes class 1 is 'malignant'
          roc_auc = auc(fpr, tpr)

          # --- Plot ROC Curve ---
          plt.figure(figsize=(6, 6))
          plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.4f})')
          plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
          plt.xlim([0.0, 1.0])
          plt.ylim([0.0, 1.05])
          plt.xlabel('False Positive Rate')
          plt.ylabel('True Positive Rate')
          plt.title(f'Receiver Operating Characteristic - Epoch {epoch}')
          plt.legend(loc="lower right")
          plt.grid(True)
          plt.tight_layout()
          plt.savefig(f'/path/to/save/roc_curve_epoch_{epoch}.png')
          plt.close()
          
          if config.wandb:
              wandb.log({
                'Test Accuracy': float(test_accuracy),
                'Test AUC ': float(test_auc_ovr),
                'Test Sensitivity': float(sensitivity),
                'Test Specificity': float(specificity),
                'FI Score': float(f1),
                'Test Loss': float(running_loss),
                'ROC Curve': wandb.Image(f'/path/to/save/roc_curve_epoch_{epoch}.png'),
                })
          model.train()
          #xxx
# def compute_accuracy(true_labels, predicted_labels):
#     correct = sum(1 for true, pred in zip(true_labels, predicted_labels) if true == pred)
#     total = len(true_labels)
#     accuracy = correct / total 
#     return accuracy

def sen(n, con_mat):

    sen = []
    # con_mat = confusion_matrix(Y_test, Y_pred)

    if n == 2:
        for i in range(1, n):
            tp = con_mat[i][i]
            fn = np.sum(con_mat[i, :]) - tp
            sen1 = tp / (tp + fn)
            sen.append(sen1)

    else:
        for i in range(n):
            tp = con_mat[i][i]
            fn = np.sum(con_mat[i, :]) - tp
            sen1 = tp / (tp + fn)
            sen.append(sen1)

    return sen

def spe(n, con_mat):
    spe = []
    # con_mat = confusion_matrix(Y_test, Y_pred)

    if n == 2:
        for i in range(1, n):
            number = np.sum(con_mat[:, :])
            tp = con_mat[i][i]
            fn = np.sum(con_mat[i, :]) - tp
            fp = np.sum(con_mat[:, i]) - tp
            tn = number - tp - fn - fp
            spe1 = tn / (tn + fp)
            spe.append(spe1)
    else:
        for i in range(n):
            number = np.sum(con_mat[:, :])
            tp = con_mat[i][i]
            fn = np.sum(con_mat[i, :]) - tp
            fp = np.sum(con_mat[:, i]) - tp
            tn = number - tp - fn - fp
            spe1 = tn / (tn + fp)
            spe.append(spe1)

    return spe




if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--wandb', type=str, default=True)  # False   True
    parser.add_argument('--run_name', type=str, default=f'ResNet152-heatmap_test3')
    parser.add_argument('--train_csv', type=str, default='/path/to/dataset/ISBI2016_ISIC_Part3_Training_GroundTruth.csv') # 
    parser.add_argument('--test_csv', type=str, default='/path/to/dataset/ISBI2016_ISIC_Part3_Test_GroundTruth.csv') # 
    parser.add_argument('--testing_dataset', type=str, default='/path/to/dataset/ISBI2016_ISIC_Part3_Test_Data') # 
    parser.add_argument('--training_dataset', type=str, default='/path/to/dataset/ISBI2016_ISIC_Part3_Training_Data') # 
    parser.add_argument('--model_path', type=str, default='/path/for/saved_model') # save models
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--batch_size', type=int, default=32)  # 48
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--num_epochs', type=int, default=402)
    parser.add_argument('--saved_epochs', type=int, default=50) # 50
    config = parser.parse_args()
    main(config)





    



