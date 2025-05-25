""" U-Net implementation in PyTorch
    This code implements the U-Net architecture for image segmentation tasks.
    It includes the following components:
    - DoubleConv: A block of two convolutional layers with batch normalization and ReLU activation.
    - DownSample: A block that downsamples the input using max pooling followed by a double convolution.
    - UpSample: A block that upsamples the input using either bilinear interpolation or transposed convolution, followed by a double convolution.
    - OutConv: A block that applies a 1x1 convolution to map the output to the desired number of classes.
    - UNet: The main U-Net architecture that combines the above components to create a full model.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """
    Double Convolution block: (Conv2d -> BatchNorm -> ReLU) × 2
    
    This block is used multiple times in the U-Net architecture.
    """
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        
        #Implementation of the double convolution block
        #sequential module with two Conv2d layers
        #each followed by BatchNorm2d and ReLU
        #first Conv2d goes from in_channels to out_channels
        #second maintains the out_channels
        
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        #forward pass of the double convolution block
        #applies the double_conv sequential to the input tensor
        
        return self.double_conv(x)


class DownSample(nn.Module):
    """
    Downsampling block: MaxPool2d followed by DoubleConv
    """
    def __init__(self, in_channels, out_channels):
        super(DownSample, self).__init__()
        
        #Implementation of the downsampling block
        #MaxPool2d with kernel_size=2 for downsampling
        #Then DoubleConv block implemented above
        
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            DoubleConv(in_channels, out_channels)
        )
    
    def forward(self, x):
        #forward pass of the downsampling block
        #applying the maxpool_conv sequential to the input tensor
        
        return self.maxpool_conv(x)


class UpSample(nn.Module):
    """
    Upsampling block: Upsample followed by DoubleConv
    """
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(UpSample, self).__init__()
        
        #Implementation of the upsampling block
        #If bilinear is True, use nn.Upsample with mode='bilinear'
        #If bilinear is False, use ConvTranspose2d for learnable upsampling
        #then reducing the number of channels by a DoubleConv
        
        if bilinear:
            self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels + in_channels // 2, out_channels)
        else:
            self.upsample = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)
            
    def forward(self, x1, x2):
        #forward pass of the upsampling block
        #First upsample x1, then handle the concatenation with x2
        #x1 and x2 may have different spatial dimensions
        #need to deal with this using operations like center crop or padding, here i use padding to preserve spatial features from the encoder
        #then applying double convolution
        
        x1 = self.upsample(x1)

        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, (diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2))
        x = torch.cat([x2, x1], dim=1)

        return self.conv(x)

class OutConv(nn.Module):
    """
    Output Convolution block: 1x1 convolution to map to the required number of classes
    """
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        
        #Implementation of the output convolution
        #using a 1x1 Conv2d to map to the required number of output classes
        
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        pass

    def forward(self, x):
        #Implementation of the forward pass of the output convolution
        #applying the conv layer to the input tensor
        
        x = self.conv(x)
        
        return x


class UNet(nn.Module):
    """
    Full U-Net architecture
    """
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        
        #Input layer
        #Implementing the input double convolution
        #using DoubleConv to process the input image
        self.inconv = DoubleConv(n_channels, 64)
        
        #Encoder (downsampling) path
        #Implementing the encoder path with multiple downsampling steps
        #creating multiple DownSample modules with increasing channel depth
        #the typical channel progression looks like: 64 -> 128 -> 256 -> 512 -> 1024
        self.down1 = DownSample(64, 128) 
        self.down2 = DownSample(128, 256)  
        self.down3 = DownSample(256, 512)  
        self.down4 = DownSample(512, 1024)  
       
        #Decoder (upsampling) path
        #Implementing the decoder path with multiple upsampling steps
        #creating multiple UpSample modules with decreasing channel depth
        #the typical channel progression looks like: 1024 -> 512 -> 256 -> 128 -> 64
        self.up1 = UpSample(1024, 512, bilinear)
        self.up2 = UpSample(512, 256, bilinear)
        self.up3 = UpSample(256, 128, bilinear)
        self.up4 = UpSample(128, 64, bilinear)
        
        #Output layer
        #Implementing the output convolution
        #using OutConv to produce the final segmentation map
        self.outconv = OutConv(64, n_classes)

    
    def forward(self, x):
        #Implementing the forward pass of the U-Net
        #FollowING the U-Net architecture diagram
        #1. Apply the input convolution
        #2. Apply encoder blocks and save the outputs for skip connections
        #3. Apply decoder blocks with skip connections
        #4. Apply the output convolution
        
        #Input convolution
        x1 = self.inconv(x)
        
        #Encoder path
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        #Decoder path with skip connections
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        
        #Output convolution
        x = self.outconv(x)
        
        return x    #return the final output tensor


'''
#testing the implementation
def test_unet():
    #create a random input tensor
    batch_size = 1
    channels = 3
    height = 572
    width = 572
    x = torch.randn(batch_size, channels, height, width)
    
    double_conv = DoubleConv(channels, 64)
    print(double_conv)

    #create the U-Net model
    model = UNet(n_channels=channels, n_classes=2, bilinear=True)
    
    #forward pass
    output = model(x)
    rm
    #check output shape
    expected_shape = (batch_size, 2, height, width)
    assert output.shape == expected_shape, f"Expected shape {expected_shape}, got {output.shape}"
    
    print(output.size())
    print("U-Net implementation test passed!")
    return output
'''

#Uncomment the lines above and below to test implementation
#test_output = test_unet()






















