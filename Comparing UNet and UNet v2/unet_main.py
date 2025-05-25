import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
from torchvision.utils import save_image
import wandb
import sys
import segmentation_models_pytorch as smp #use smp for pretrained unet

from unet_model import UNet
from segmentation_dataset import lesion_segmentation_dataset


#Configuration
train_image_dir = '/path/to/train/images'
train_mask_dir = '/path/to/train/masks'
test_image_dir = '/path/to/test/images'
test_mask_dir = '/path/to/test/masks'
val_image_dir = '/path/to/val/images'
val_mask_dir = '/path/to/val/masks'
predicted_mask_dir = '/path/to/save/predictions/unet_predictions' #path to save prediction masks

batch_size = 32
epochs = 300
lr = 1e-3
image_size = (256, 256)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


#Dataset loading and preparation
train_dataset = lesion_segmentation_dataset(train_image_dir, train_mask_dir, image_size, augment=True, normalize=True)
val_dataset = lesion_segmentation_dataset(val_image_dir, val_mask_dir, image_size, augment=False, normalize=True)
test_dataset = lesion_segmentation_dataset(test_image_dir, test_mask_dir, image_size, augment=False, normalize=True)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)


#Weights and Biases (wandb) initialization
wandb.init(
    id=wandb.util.generate_id(),
    project='UNet-ISIC17-segmentation',
    name='UNet_Segmentation',
    reinit=True,
    config={
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "image_size": image_size,
        "bce_weight": 0.2,
        "dice_weight": 0.8,
        "optimizer": "Adam",
        "lr_scheduler": "Polynomial",
        "model": "UNet",
        "loss_function": "BCE + Dice (Dynamic)",
        "augmentation": True,
        "model_architecture": "U-Net",
        "pretrained": True,
    },
    settings=wandb.Settings(_executable='<executable>'),
)

#Loss Function
class BCEDiceLoss(nn.Module):
    def __init__(self, bce_weight=0.5, dice_weight=0.5):
        super(BCEDiceLoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

    def forward(self, inputs, targets, eps=1e-6):
        bce_loss = self.bce(inputs, targets)

        inputs = torch.sigmoid(inputs) #sigmoid for binary classification
        targets = targets.float()

        intersection = (inputs * targets).sum(dim=(1,2,3))
        union = inputs.sum(dim=(1,2,3)) + targets.sum(dim=(1,2,3))
        dice = (2. * intersection + eps) / (union + eps)
        dice_loss = 1 - dice.mean()

        return self.bce_weight * bce_loss + self.dice_weight * dice_loss


#Model initialization *Use comment below for vanilla, not-pretrained U-Net*
#model = UNet(n_channels=3, n_classes=1, bilinear=True).to(device)
model = smp.Unet(
    encoder_name="resnet34",
    encoder_weights="imagenet",
    in_channels=3,
    classes=1,
    bilinear=True,
).to(device)

criterion = BCEDiceLoss(bce_weight=0.2, dice_weight=0.8)
optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))


#Helper functions *normally commented out when not in use
class EarlyStopping:
    def __init__(self, patience=20, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

early_stopper = EarlyStopping(patience=25, min_delta=1e-4)


def get_dynamic_weights(epoch, total_epochs):
    bce_weight = max(0.1, 1 - epoch / total_epochs) 
    dice_weight = 1.0 - bce_weight                  
    return bce_weight, dice_weight




#Learning Rate Scheduler
def adjust_polynomial_lr(optimizer, epoch, max_epochs, initial_lr, power=0.9, warmup_epochs=5):
    if epoch < warmup_epochs:

        new_lr = initial_lr * (epoch + 1) / warmup_epochs
    else:
        decay_epoch = epoch - warmup_epochs
        decay_total = max_epochs - warmup_epochs
        new_lr = initial_lr * (1 - decay_epoch / decay_total) ** power

    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr
    return new_lr


#--------------------------------------------------------------------------------
#training
train_losses = []
val_losses = []
lr_decay = []

for epoch in range(epochs):
    model.train()
    running_loss = 0.0

    current_lr = adjust_polynomial_lr(optimizer, epoch, epochs, lr, power=0.9,warmup_epochs=5)
    lr_decay.append(current_lr)

    #dynamic weighting 
    #bce, dice = get_dynamic_weights(epoch, epochs)
    #criterion = BCEDiceLoss(bce_weight=bce, dice_weight=dice)

    for images, masks in train_loader:
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_train_loss = running_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    #Validation
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for val_images, val_masks in val_loader:
            val_images = val_images.to(device)
            val_masks = val_masks.to(device)
            val_outputs = model(val_images)
            loss = criterion(val_outputs, val_masks)
            val_loss += loss.item()
    avg_val_loss = val_loss / len(val_loader)
    val_losses.append(avg_val_loss)

    print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}")

    wandb.log({
        "Training Loss": avg_train_loss,
        "Validation Loss": avg_val_loss,
        "Epoch": epoch + 1
    })


    #early_stopper(avg_val_loss)
    #if early_stopper.early_stop:
    #    print(f"Early stopping triggered at epoch {epoch+1}")
    #    break


#--------------------------------------------------------------------------------
#Metric Calculation
def calculate_metrics(pred, target, eps=1e-6):
    pred = (torch.sigmoid(pred) > 0.5).float()
    target = target.float()

    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()

    iou = intersection / (union - intersection + eps)
    dice = 2 * intersection / (union + eps)

    return iou.item(), dice.item()



#Denormalization function to convert tensor back to original image scale
def denormalize(tensor, mean, std):
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return torch.clamp(tensor, 0, 1)




#--------------------------------------------------------------------------------
#Test Loop
model.eval()
os.makedirs(predicted_mask_dir, exist_ok=True) #directory to save predicted masks
total_iou, total_dice = 0.0, 0.0

with torch.no_grad():
    for i, (image, mask) in enumerate(test_loader):
        image = image.to(device)
        mask = mask.to(device)

        output = model(image)
        pred_mask = torch.sigmoid(output)
        binarized = (pred_mask > 0.45).float()


        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
       
        image_cpu = image.squeeze(0).detach().cpu()
        pred_cpu = binarized.squeeze(0).detach().cpu()
        mask_cpu = mask.squeeze(0).detach().cpu()

        denorm_image = denormalize(image_cpu.squeeze(0).cpu().clone(), mean, std)

        #Save Triplet for Qualitative Metric: [Test Image, Predicted Mask, Groundtruth Mask]
        stacked = torch.stack([denorm_image, pred_cpu.repeat(3,1,1), mask_cpu.repeat(3,1,1)])
        save_path = os.path.join(predicted_mask_dir, f"triplet_{i:03}.png")
        save_image(stacked, save_path, nrow=3)

        iou, dice = calculate_metrics(output, mask)
        total_iou += iou
        total_dice += dice


#Final Metric Calculation
num_samples = len(test_loader)
final_iou = total_iou / num_samples
final_dice = total_dice / num_samples


wandb.log({
    "Final IoU": final_iou,
    "Final Dice": final_dice,
})

print(f"\nFinal IoU score: {final_iou:.4f}")
print(f"Final Dice Coefficient score: {final_dice:.4f}")

#Plotting
epochs_trained = len(train_losses)
plt.figure(figsize=(10, 5))
plt.plot(range(epochs_trained), train_losses, label='Training Loss', color='blue')
plt.plot(range(epochs_trained), val_losses, label='Validation Loss', color='orange')
plt.title('Training and Validation Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)
plt.legend()
plt.savefig('/path/to/save/figures/unet_training_validation.png')
plt.show()

wandb.finish()
sys.exit(0)