# importing all the necessary libraries
import torch.nn as nn
import torch.nn.functional as F

class Conv2D(nn.Module):
    def __init__(self, in_channels, out_channels, equal=False):
        super().__init__()
        # Convolutional layer
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)
        # Batch normalization layer
        self.bn = nn.BatchNorm2d(out_channels)
        # Activation function (LeakyReLU)
        self.act = nn.LeakyReLU(0.1)
    
    def forward(self, x):
        x = self.conv(x)  # Convolutional operation
        x = self.bn(x)  # Batch normalization
        x = self.act(x)  # Activation function
        return x

class ConvTrans2D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # Transposed convolutional layer
        self.convT = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)
        # Batch normalization layer
        self.bn = nn.BatchNorm2d(out_channels)
        # Activation function (LeakyReLU)
        self.act = nn.LeakyReLU(0.1)
    
    def forward(self, x):
        x = self.convT(x)  # Transposed convolutional operation
        x = self.bn(x)  # Batch normalization
        x = self.act(x)  # Activation function
        return x
    
class ResNet(nn.Module):
    def __init__(self, channels):
        super().__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        # Batch normalization layer
        self.bn = nn.BatchNorm2d(channels)
        
    def forward(self, x):
        residual = x  # Store the input as the residual
        x = self.conv1(x)  # Convolutional operation
        x = self.bn(x)  # Batch normalization
        x = F.relu(x)  # ReLU activation function
        x = self.conv2(x)  # Convolutional operation
        x = x + residual  # Add the residual to the output
        return x


import os
import torch
import torch.nn as nn

# Gan generator 
class Generator(nn.Module):
    def __init__(self, dimension, spatial_channel):
        super().__init__()
        
        self.dimension = dimension
        self.spatial_channel = spatial_channel
        
        self.encoder_channels = [self.spatial_channel, 56, 112, 224, 448]
        self.encoder = nn.ModuleList()
        for i in range(1, len(self.encoder_channels)):      # making an encoder layers and appending all layers to a list
            self.encoder.append(Conv2D(self.encoder_channels[i-1], self.encoder_channels[i]))
        
        self.resnet = nn.ModuleList()
        self.n_resnet = 9
        for i in range(self.n_resnet):  # appending the list with resnet layers
            self.resnet.append(ResNet(self.encoder_channels[-1]))
        
        self.decoder_channels = [self.encoder_channels[-1], 224, 112, 56, 3]    
        self.decoder = nn.ModuleList()
        for i in range(1, len(self.decoder_channels)): #appending the list with convTrans layers for decoding
            self.decoder.append(ConvTrans2D(self.decoder_channels[i-1], self.decoder_channels[i]))
        
    def forward(self, x):
        for i in range(len(self.encoder)):
            x = self.encoder[i](x)      # first the input will be encoded by passing through the encoder layers
        for i in range(len(self.resnet)):
            x = self.resnet[i](x)       # then we have the resnet layers to reduce the vanishing gradient 
        for i in range(len(self.decoder)):
            x = self.decoder[i](x)      # lastly decoding the image to form the generated image
        x = torch.sigmoid(x)
        return x

# discrimator class to discrimate between the real image and the generated image         
class Discriminator(nn.Module):
    def __init__(self, dimension, spatial_channel, avgpool):
        super().__init__()
        
        self.dimension = dimension
        self.spatial_channel = spatial_channel
        self.avgpool = avgpool
        # Create a list of average pooling layers
        self.pool = nn.ModuleList()
        for i in range(avgpool):
            self.pool.append(nn.AvgPool2d(kernel_size=4, stride=2, padding=1))
        # Define the number of channels for the discriminator
        self.dis_channels = [self.spatial_channel + 3, 64, 128, 256, 512]
        # Create a list of convolutional layers for the discriminator
        self.dis = nn.ModuleList()
        for i in range(1, len(self.dis_channels)):
            self.dis.append(nn.Conv2d(self.dis_channels[i-1], self.dis_channels[i], kernel_size=4))
            
    def forward(self, x):
        # Apply average pooling layers to downsample the input tensor
        for i in range(len(self.pool)):
            x = self.pool[i](x)
        # Apply convolutional layers to extract features from the downsampled tensor
        for i in range(len(self.dis)):
            x = self.dis[i](x)        
        # Apply sigmoid activation function to the final output tensor
        x = torch.sigmoid(x)
        return x

    
class GanModule(nn.Module):
    
    def __init__(self, generator=True, discriminator=True):
        super().__init__()
        self.G = None
        self.D1 = None
        self.D2 = None
        self.D3 = None
        
        self.dimension = 512
        self.spatial_channel = 32
        
        if generator:
            self.G = Generator(self.dimension, self.spatial_channel)
        
        if discriminator:
            self.D1 = Discriminator(self.dimension, self.spatial_channel, avgpool=0)
            self.D2 = Discriminator(self.dimension, self.spatial_channel, avgpool=1)
            self.D3 = Discriminator(self.dimension, self.spatial_channel, avgpool=2)
            self.label_real = 1
            self.label_fake = 0
            
     
    def generate(self, spatial_map):
        # Generate photo using the generator network
        photo = self.G(spatial_map)
        return photo
    
    def discriminate(self, spatial_map, photo):
        # Concatenate spatial_map and photo along the channel dimension
        spatial_map_photo = torch.cat((spatial_map, photo), 1)
        
        # Pass the concatenated input through each discriminator network
        patch_D1 = self.D1(spatial_map_photo)
        patch_D2 = self.D2(spatial_map_photo)
        patch_D3 = self.D3(spatial_map_photo)
        
        return patch_D1, patch_D2, patch_D3
    
    



import torch
class BCE:
    def __init__(self):
        self.criterion = torch.nn.BCELoss()  # Binary Cross Entropy loss criterion

    def compute(self, prediction, ground_truth):
        return self.criterion(prediction, ground_truth)  # Compute BCE loss


class MSE:
    def __init__(self):
        self.criterion = torch.nn.MSELoss()  # Mean Squared Error loss criterion

    def compute(self, prediction, ground_truth):
        return self.criterion(prediction, ground_truth)  # Compute MSE loss


class L1:
    def __init__(self):
        self.criterion = torch.nn.L1Loss()  # L1 loss criterion

    def compute(self, prediction, ground_truth):
        return self.criterion(prediction, ground_truth)  # Compute L1 loss
    
    
import torch
from torchvision import transforms
import gc
torch.cuda.empty_cache()
gc.collect()

class Perceptual:
    # Preprocess transformation for input images
    preprocess = transforms.Compose([
        transforms.Resize(256),                         # Resize images to 256x256
        transforms.CenterCrop(224),                     # Center crop to 224x224
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize using mean and standard deviation
    ])

    perceptual_layer = ['4', '9', '14', '19']           # List of perceptual layers

    def __init__(self, device='cpu'):
        self.device = torch.device(device)
        self.model = torch.hub.load('pytorch/vision:v0.10.0', 'vgg11', pretrained=True)  # Load pre-trained VGG model
        self.model.to(self.device)
        self.criterion = L1()                            # L1 loss criterion
    
    def compute(self, prediction, ground_truth):
        prediction = self.preprocess(prediction)         # Preprocess prediction image
        ground_truth = self.preprocess(ground_truth)     # Preprocess ground truth image
        
        loss = 0
        for layer, module in self.model.features._modules.items():
            prediction = module(prediction)               # Pass prediction through VGG model
            ground_truth = module(ground_truth)           # Pass ground truth through VGG model
            if layer in self.perceptual_layer:            # Check if current layer is a perceptual layer
                loss += self.criterion.compute(prediction, ground_truth)  # Compute L1 loss between prediction and ground truth at the perceptual layer
        return loss
