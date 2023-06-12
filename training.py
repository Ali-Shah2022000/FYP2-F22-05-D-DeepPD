# !export PYTORCH_cpu_ALLOC_CONF=max_split_size_mb=512
import glob
import cv2
import numpy as np
import  copy
import torch
import torch.nn.functional as F
# import mlflow




def segmentImage(anotImages,hedAfterSimo,i,allFmOuputs):  # function to create segments
        import torch
        print(i)
        merged = torch.zeros((1, 32, 512, 512))

        fullImage = cv2.imread(hedAfterSimo[i])      # read the full image
        # cv2.imwrite("FeatureMapping/1.jpg",fullImage,)  #saving the image

#         print(fullImage.shape)
        remainingFace = copy.deepcopy(fullImage)   # making a deep copy of the full image so that it does not change
        dimColoredImage = (fullImage.shape[0], fullImage.shape[1])   # get the dimensions of the image

        img = cv2.imread(anotImages[i][0],cv2.IMREAD_GRAYSCALE)         # read the anot Image in grayscale form,
        img = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)       # make a zero array of the image
        segmentedImages = [img, img, img, img]          # make 4 copies of the zero array , this will store all the segments

        for j in anotImages[i]:     # loop to store the segments
            attribute = j[j.index("_") + 1:j.rindex(".")]       # get the attribite name from the anot image
            if attribute == "r_brow" or attribute =="r_eye":                # from our point left side
                blackAndWhiteImg = cv2.imread(j)            # reading the anot image
                segmentedImages[0] = cv2.bitwise_or(segmentedImages[0],blackAndWhiteImg)        # bitwise or to save the eyebrow and the eye

            elif attribute =="l_brow" or attribute == "l_eye":      # for the left eye
                blackAndWhiteImg = cv2.imread(j)
                segmentedImages[1] = cv2.bitwise_or(segmentedImages[1], blackAndWhiteImg)

            elif attribute =="nose" :
                blackAndWhiteImg = cv2.imread(j)
                segmentedImages[2] = cv2.bitwise_or(segmentedImages[2], blackAndWhiteImg)

            elif attribute == "mouth" or attribute ==" l_lip"  or attribute == "u_lip":
                blackAndWhiteImg = cv2.imread(j)
                segmentedImages[3] = cv2.bitwise_or(segmentedImages[3], blackAndWhiteImg)


        merged = allFmOuputs[4]

        # k==3 mouth
        segmentedImages[3] = cv2.resize(segmentedImages[3], dimColoredImage, interpolation=cv2.INTER_AREA)      # resizing the images to 512
        gray = cv2.cvtColor(segmentedImages[3], cv2.COLOR_BGR2GRAY)         # converting the image to gray scale
        contours, _ = cv2.findContours(gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)        # get the contours of the iamge
        x, y, w, h = cv2.boundingRect(contours[0])
        crop_img = fullImage[y - 40:y + h + 70, x - 25:x + w + 25]           # cropping the image (manually set)
        allFmOuputs[3] = F.interpolate(allFmOuputs[3], size=(crop_img.shape[0], crop_img.shape[1]), mode='bilinear')
        merged[:,:32,y - 40:y + h + 70, x - 25:x + w + 25  ]=allFmOuputs[3][:,:32,:,:]
        
        #k==2 nose
        segmentedImages[2] = cv2.resize(segmentedImages[2], dimColoredImage, interpolation=cv2.INTER_AREA)      # resizing the images to 512
        gray = cv2.cvtColor(segmentedImages[2], cv2.COLOR_BGR2GRAY)         # converting the image to gray scale
        contours, _ = cv2.findContours(gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)        # get the contours of the iamge
        x, y, w, h = cv2.boundingRect(contours[0])
        crop_img = fullImage[y - 5:y + h + 35, x - 20:x + w + 20]           # cropping the image (manually set)
        allFmOuputs[2] = F.interpolate(allFmOuputs[2], size=(crop_img.shape[0], crop_img.shape[1]), mode='bilinear')
        merged[:,:32,y - 5:y + h + 35, x - 20:x + w + 20]=allFmOuputs[2][:,:32,:,:]
        
        #k==1 r_eye
        segmentedImages[1] = cv2.resize(segmentedImages[1], dimColoredImage, interpolation=cv2.INTER_AREA)      # resizing the images to 512
        gray = cv2.cvtColor(segmentedImages[1], cv2.COLOR_BGR2GRAY)         # converting the image to gray scale
        contours, _ = cv2.findContours(gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)        # get the contours of the iamge
        x, y, w, h = cv2.boundingRect(contours[0])
        crop_img = fullImage[y - 50:y + h + 30, x - 30:x + w + 70]        # cropping the image (manually set)
        allFmOuputs[1] = F.interpolate(allFmOuputs[1], size=(crop_img.shape[0], crop_img.shape[1]), mode='bilinear')
        merged[:,:32,y - 50:y + h + 30, x - 30:x + w + 70]=allFmOuputs[1][:,:32,:,:]


         #k==0 l_eye
        segmentedImages[0] = cv2.resize(segmentedImages[0], dimColoredImage, interpolation=cv2.INTER_AREA)      # resizing the images to 512
        gray = cv2.cvtColor(segmentedImages[0], cv2.COLOR_BGR2GRAY)         # converting the image to gray scale
        contours, _ = cv2.findContours(gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)        # get the contours of the iamge
        x, y, w, h = cv2.boundingRect(contours[0])
        crop_img = fullImage[y - 50:y + h + 30, x - 70:x + w + 30]         # cropping the image (manually set)
        allFmOuputs[0] = F.interpolate(allFmOuputs[0], size=(crop_img.shape[0], crop_img.shape[1]), mode='bilinear')
        merged[:,:32,y - 50:y + h + 30, x - 70:x + w + 30]=allFmOuputs[0][:,:32,:,:]


        return merged

        
        
        # image = merged.squeeze().detach().numpy()
        # cv2.imshow("Image", image[0])
        # # Wait for a key press and then close the window
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()


import torch        
from torch import nn


# device ="cpu"
# if torch.cpu.is_available():
#     device ="cpu"

def define_part_encoder(model='eye', norm='instance', input_nc=1, latent_dim=512):
    # Set the normalization layer based on the given 'norm' parameter
    norm_layer = nn.BatchNorm2d

    # Set the default image size to 512
    image_size = 512

    # Update the image size based on the 'model' parameter
    if 'eye' in model:
        image_size = 128
    elif 'mouth' in model:
        image_size = 192
    elif 'nose' in model:
        image_size = 160
    elif 'face' in model:
        image_size = 512

    # Create an instance of the EncoderGenerator_Res class with the specified parameters
    net_encoder = EncoderGenerator_Res(norm_layer, image_size, input_nc, latent_dim)

    # Return the created encoder network
    return net_encoder

class EncoderBlock(nn.Module):
    def __init__(self, channel_in, channel_out, kernel_size=7, padding=3, stride=4):
        super(EncoderBlock, self).__init__()
        # Define a convolutional layer with the specified number of input and output channels,
        # kernel size, padding, and stride.
        self.conv = nn.Conv2d(channel_in, channel_out, kernel_size, padding=padding, stride=stride)
        # Define a batch normalization layer with the specified number of output channels.
        self.bn = nn.BatchNorm2d(channel_out, momentum=0.9)
        # Define a LeakyReLU activation function with a negative slope of 1.
        self.relu = nn.LeakyReLU(1)

    def forward(self, ten, out=False, t=False):
        if out:
            # If 'out' is True, apply the convolutional layer to the input tensor.
            ten = self.conv(ten)
            # Save the output tensor before batch normalization and activation.
            ten_out = ten
            # Apply batch normalization to the tensor.
            ten = self.bn(ten)
            # Apply the LeakyReLU activation function to the tensor.
            ten = self.relu(ten)
            # Return both the processed tensor and the saved output tensor.
            return (ten, ten_out)
        else:
            # If 'out' is False, apply the convolutional layer to the input tensor.
            ten = self.conv(ten)
            # Apply batch normalization to the tensor.
            ten = self.bn(ten)
            # Apply the LeakyReLU activation function to the tensor.
            ten = self.relu(ten)
            # Return the processed tensor.
            return ten
    

class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, activation=nn.ReLU(True), use_dropout=False):
        super(ResnetBlock, self).__init__()
        # Build the convolutional block for the ResNet block
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, activation, use_dropout)
        # Move the conv_block to the GPU if available
        self.conv_block # move to GPU

    def build_conv_block(self, dim, padding_type, norm_layer, activation, use_dropout):
        conv_block = []
        p = 0
        # Check the padding type and add the corresponding padding layer to the conv_block
        if (padding_type == 'reflect'):
            conv_block += [nn.ReflectionPad2d(1)]
        elif (padding_type == 'replicate'):
            conv_block += [nn.ReplicationPad2d(1)]
        elif (padding_type == 'zero'):
            p = 1
        # Add a convolutional layer, normalization layer, and activation function to the conv_block
        conv_block += [nn.Conv2d(dim, dim, 3, padding=p), norm_layer(dim), activation]
        # Add a dropout layer if use_dropout is True
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]
        p = 0
        # Check the padding type again and add the corresponding padding layer to the conv_block
        if (padding_type == 'reflect'):
            conv_block += [nn.ReflectionPad2d(1)]
        elif (padding_type == 'replicate'):
            conv_block += [nn.ReplicationPad2d(1)]
        elif (padding_type == 'zero'):
            p = 1
        # Add a second convolutional layer and normalization layer to the conv_block
        conv_block += [nn.Conv2d(dim, dim, 3, padding=p), norm_layer(dim)]
        # Create a sequential module using the conv_block list
        return nn.Sequential(*conv_block)

    def forward(self, x):
        # Apply the ResNet block by adding the input tensor to the conv_block output
        out = (x + self.conv_block(x))
        return out

class EncoderGenerator_Res(nn.Module):
    def __init__(self, norm_layer, image_size, input_nc, latent_dim=512):
        super(EncoderGenerator_Res, self).__init__()

        layers_list = []

        latent_size = int(image_size / 32)
        longsize = 512 * latent_size * latent_size
        self.longsize = longsize

        activation = nn.ReLU()
        padding_type = 'reflect'
        norm_layer = nn.BatchNorm2d

        # Add the initial EncoderBlock to the layers_list
        layers_list.append(
            EncoderBlock(channel_in=input_nc, channel_out=32, kernel_size=4, padding=1, stride=2))  # 176 176

        dim_size = 32
        for i in range(4):
            # Add a ResnetBlock to the layers_list
            layers_list.append(
                ResnetBlock(dim_size, padding_type=padding_type, activation=activation, norm_layer=norm_layer))
            # Add another EncoderBlock to the layers_list
            layers_list.append(
                EncoderBlock(channel_in=dim_size, channel_out=dim_size * 2, kernel_size=4, padding=1, stride=2))
            dim_size *= 2

        # Add a final ResnetBlock to the layers_list
        layers_list.append(ResnetBlock(512, padding_type=padding_type, activation=activation, norm_layer=norm_layer))
        # Create a sequential module using the layers_list
        self.conv = nn.Sequential(*layers_list)
        # Add a fully connected layer for the mean (mu) calculation
        self.fc_mu = nn.Sequential(nn.Linear(in_features=longsize, out_features=latent_dim))

    def forward(self, ten):
        # Apply the convolutional layers to the input tensor
        ten = self.conv(ten)
        # Reshape the tensor to have a flattened shape
        ten = torch.reshape(ten, [ten.size()[0], -1])
        # Pass the flattened tensor through the fully connected layer to obtain the mean (mu)
        mu = self.fc_mu(ten)
        # Return the mean (mu)
        return mu


# ============================================================================================


# ============================================================================================
class DecoderBlock(nn.Module):
    def __init__(self, channel_in, channel_out, kernel_size=4, padding=1, stride=2, output_padding=0, norelu=False):
        super(DecoderBlock, self).__init__()

        layers_list = []
        # Add a transposed convolutional layer to the layers_list
        layers_list.append(nn.ConvTranspose2d(channel_in, channel_out, kernel_size, padding=padding, stride=stride,
                                              output_padding=output_padding))
        # Add a batch normalization layer to the layers_list
        layers_list.append(nn.BatchNorm2d(channel_out, momentum=0.9))
        if (norelu == False):
            # Add a LeakyReLU activation function to the layers_list if norelu is False
            layers_list.append(nn.LeakyReLU(1))
        # Create a sequential module using the layers_list
        self.conv = nn.Sequential(*layers_list)

    def forward(self, ten):
        # Apply the convolutional layers to the input tensor
        ten = self.conv(ten)
        # Return the processed tensor
        return ten

    

def define_feature_decoder(model='mouth', norm='instance', output_nc=1, latent_dim=512):
    norm_layer = nn.BatchNorm2d

    image_size = 512
    if 'eye' in model:
        image_size = 128
    elif 'mouth' in model:
        image_size = 192
    elif 'nose' in model:
        image_size = 160
    else:
        print("Whole Image !!")

    # Create an instance of the DecoderGenerator_feature_Res using the specified parameters
    net_decoder = DecoderGenerator_feature_Res(norm_layer, image_size, output_nc, latent_dim) # input longsize 256 to 512*4*4

    print("net_decoder to image of part " + model + " is:", image_size)

    return net_decoder


class DecoderGenerator_feature_Res(nn.Module):
    def __init__(self, norm_layer, image_size, output_nc, latent_dim=512):
        super(DecoderGenerator_feature_Res, self).__init__()

        latent_size = int(image_size/32)
        self.latent_size = latent_size
        longsize = 512*latent_size*latent_size

        activation = nn.ReLU()
        padding_type='reflect'
        norm_layer=nn.BatchNorm2d

        # Create a fully connected layer to convert the latent vector to the appropriate size
        self.fc = nn.Sequential(nn.Linear(in_features=latent_dim, out_features=longsize))

        layers_list = []

        # Add a ResnetBlock to the layers_list
        layers_list.append(ResnetBlock(512, padding_type=padding_type, activation=activation, norm_layer=norm_layer))  # 176 176
        # Add a DecoderBlock to the layers_list
        layers_list.append(DecoderBlock(channel_in=512, channel_out=256, kernel_size=4, padding=1, stride=2, output_padding=0)) # 22 22
        layers_list.append(ResnetBlock(256, padding_type=padding_type, activation=activation, norm_layer=norm_layer))  # 176 176
        # Add a DecoderBlock to the layers_list
        layers_list.append(DecoderBlock(channel_in=256, channel_out=256, kernel_size=4, padding=1, stride=2, output_padding=0)) # 44 44
        layers_list.append(ResnetBlock(256, padding_type=padding_type, activation=activation, norm_layer=norm_layer))  # 176 176
        # Add a DecoderBlock to the layers_list
        layers_list.append(DecoderBlock(channel_in=256, channel_out=128, kernel_size=4, padding=1, stride=2, output_padding=0)) # 88 88
        layers_list.append(ResnetBlock(128, padding_type=padding_type, activation=activation, norm_layer=norm_layer))  # 176 176
        # Add a DecoderBlock to the layers_list
        layers_list.append(DecoderBlock(channel_in=128, channel_out=64, kernel_size=4, padding=1, stride=2, output_padding=0)) # 176 176
        layers_list.append(ResnetBlock(64, padding_type=padding_type, activation=activation, norm_layer=norm_layer))  # 176 176
        # Add a DecoderBlock to the layers_list
        layers_list.append(DecoderBlock(channel_in=64, channel_out=64, kernel_size=4, padding=1, stride=2, output_padding=0)) # 352 352
        layers_list.append(ResnetBlock(64, padding_type=padding_type, activation=activation, norm_layer=norm_layer))  # 176 176
        # Add a reflection padding layer to the layers_list
        layers_list.append(nn.ReflectionPad2d(2))
        # Add a convolutional layer to the layers_list
        layers_list.append(nn.Conv2d(64, output_nc, kernel_size=5, padding=0))
        self.conv = nn.Sequential(*layers_list)

    def forward(self, ten):
        # Apply the fully connected layer to the input tensor
        ten = self.fc(ten)        
        # Reshape the tensor to the desired size
        ten = torch.reshape(ten, (ten.size()[0], 512, self.latent_size, self.latent_size))
        # Apply the convolutional layers to the tensor
        ten = self.conv(ten)    
        return ten




class featureMapping(nn.Module):
    def __init__(self, part):
        super(featureMapping, self).__init__()
        self.featureModel = define_feature_decoder(part, output_nc=32)

    def forward(self, x):
        # encoded = self.encodermodel(x)
        featureDecoded = self.featureModel(x)
        return featureDecoded

def define_G(input_nc, output_nc, ngf, n_downsample_global=3, n_blocks_global=9, norm='instance'):
    norm_layer = nn.BatchNorm2d    
    netG = GlobalGenerator(input_nc, output_nc, ngf, n_downsample_global, n_blocks_global, norm_layer)
    return netG

class GlobalGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, n_downsampling=3, n_blocks=9, norm_layer=nn.BatchNorm2d, padding_type='reflect'):
        assert (n_blocks >= 0)
        super(GlobalGenerator, self).__init__()

        activation = nn.ReLU()

        # Initial convolutional layer
        model = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf, 7, padding=0), norm_layer(ngf), activation]

        # Downsampling layers
        for i in range(n_downsampling):
            mult = (2 ** i)
            model += [nn.Conv2d((ngf * mult), ((ngf * mult) * 2), 3, stride=2, padding=1), norm_layer(((ngf * mult) * 2)), activation]

        # Residual blocks
        mult = (2 ** n_downsampling)
        for i in range(n_blocks):
            model += [ResnetBlock((ngf * mult), padding_type=padding_type, activation=activation, norm_layer=norm_layer)]

        # Upsampling layers
        for i in range(n_downsampling):
            mult = (2 ** (n_downsampling - i))
            model += [nn.ConvTranspose2d((ngf * mult), int(((ngf * mult) / 2)), 3, stride=2, padding=1, output_padding=1), norm_layer(int(((ngf * mult) / 2))), activation]

        # Final convolutional layer
        model += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, 7, padding=0), nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        return self.model(input)




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
        self.criterion = torch.nn.BCELoss()

    def compute(self, prediction, ground_truth):
        return self.criterion(prediction, ground_truth)
    
class MSE:
    def __init__(self):
        self.criterion = torch.nn.MSELoss()
    
    def compute(self, prediction, ground_truth):
        return self.criterion(prediction, ground_truth)


class L1:
    def __init__(self):
        self.criterion = torch.nn.L1Loss()

    def compute(self, prediction, ground_truth):
        return self.criterion(prediction, ground_truth)

import torch
from torchvision import transforms

class Perceptual:
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    perceptual_layer = ['4', '9', '14', '19']

    def __init__(self, device='cpu'):
        self.device = torch.device(device)
        self.model = torch.hub.load('pytorch/vision:v0.10.0', 'vgg11', pretrained=False)
        self.model.to(self.device)
        self.criterion = L1()
    
    def compute(self, prediction, ground_truth):
        prediction = self.preprocess(prediction)
        ground_truth = self.preprocess(ground_truth)
        
        loss = 0
        for layer, module in self.model.features._modules.items():
            prediction = module(prediction)
            ground_truth = module(ground_truth)
            if layer in self.perceptual_layer:
                loss += self.criterion.compute(prediction, ground_truth)
        return loss
    

    import glob
import sys

import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt




def getAnotDict(anotPath):  # function to get all annotation images in a dictionary
    anotImages = glob.glob(f"{anotPath}/*.png")  # get all images
    anotImagesDict = {}     # create a empty dictionary
    for image in anotImages:
        oneImage = image[image.rindex("\\") + 1:]      # get the key (it will be a int number) +extension
        if int(oneImage[:oneImage.index("_")]) not in anotImagesDict:   # get the index and checking if it already exists
            anotImagesDict[int(oneImage[:oneImage.index("_")])] = []        # if not then create an empty list for that image
        anotImagesDict[int(oneImage[:oneImage.index("_")])].append(image)   # add anot image to that list with index
    return anotImagesDict

def resizedHed(hedPath):    # function to get all the images
    mainImages = glob.glob(rf"{hedPath}/*.jpg")     # getting all images names and paths
    allImages = {}  # create a empty dictionary
    for image in mainImages:
        index = int(image[image.rindex("\\") + 1:-4])   # get the index of the image (number)
        allImages[index] = image       # for the index, save the complete path of the image
    return allImages


anotImages=None
hedAfterSimo = None


def getSegments(Path):
    segments = glob.glob(Path)
    segmentsDict = {}
    for image in segments:
        index = int(image[image.rindex("\\") + 1:-4])
        segmentsDict[index] = image
    return segmentsDict


anotImages=getAnotDict(r"static\AnotImages")
print(len(anotImages))
hedAfterSimo=resizedHed(r"static\sketches(WhiteBackground)\sketches(WhiteBackground)\train\train")
print(len(hedAfterSimo))


realImages = resizedHed(r"static\hedAfterSimo")


l_eyeSegments=getSegments(r"static\Segments\Segments\train/l_eye/l_eye/*.jpg")
r_eyeSegments=getSegments(r"static\Segments\Segments\train/r_eye/r_eye/*.jpg")
noseSegments=getSegments(r"static\Segments\Segments\train/nose/nose/*.jpg")
mouthSegments=getSegments(r"static\Segments\Segments\train/mouth/mouth/*.jpg")
remainingSegments=getSegments(r"static\Segments\Segments\train/remaining/remaining/*.jpg")


l_eyeEncoder = define_part_encoder("eye")
l_eyeEncoder.load_state_dict(torch.load(r"static/Models/l_eye.pt"))


r_eyeEncoder = define_part_encoder("eye")
r_eyeEncoder.load_state_dict(torch.load(r"static/Models/r_eye.pt"))


noseEncoder = define_part_encoder("nose")
noseEncoder.load_state_dict(torch.load(r"static/Models/nose.pt"))


mouthEncoder = define_part_encoder("mouth")
mouthEncoder.load_state_dict(torch.load(r"static/Models/mouth.pt"))


remainingEncoder = define_part_encoder("face")
remainingEncoder.load_state_dict(torch.load(r"static/Models/remaining.pt"))



l_eyeFM =featureMapping("eye")
l_eyeFM.load_state_dict(torch.load(r"static/Models/l_eyeFM-5-30000.pth",map_location=torch.device('cpu')))


r_eyeFM =featureMapping("eye")
r_eyeFM.load_state_dict(torch.load(r"static/Models/r_eyeFM-5-30000.pth",map_location=torch.device('cpu')))

noseFM =featureMapping("nose")
noseFM.load_state_dict(torch.load(r"static/Models/noseFM-5-30000.pth",map_location=torch.device('cpu')))

mouthFM =featureMapping("mouth")
mouthFM.load_state_dict(torch.load(r"static/Models/mouthFM-5-30000.pth",map_location=torch.device('cpu')))

remainingFM =featureMapping("remaining")
remainingFM.load_state_dict(torch.load(r"static/Models/remainingFM-5-30000.pth",map_location=torch.device('cpu')))


transformFun = transforms.Compose([
        transforms.Grayscale(),     # converting to grayscale, so that we can get 1 channel
        transforms.ToTensor()   # converting to grayscale
    ])


imageGan = GanModule().to("cpu")
imageGan.G.load_state_dict(torch.load(r"static/Models/generator-5-30000.pth", map_location=torch.device('cpu')))
imageGan.D1.load_state_dict(torch.load(r"static/Models/D1-5-30000.pth", map_location=torch.device('cpu')))
imageGan.D2.load_state_dict(torch.load(r"static/Models/D2-5-30000.pth", map_location=torch.device('cpu')))
imageGan.D3.load_state_dict(torch.load(r"static/Models/D3-5-30000.pth", map_location=torch.device('cpu')))



label_real = imageGan.label_real
label_fake = imageGan.label_fake

optimizer_generator = torch.optim.Adam( list(l_eyeFM.parameters()) +list(r_eyeFM.parameters()) +list(noseFM.parameters()) +list(mouthFM.parameters()) +list(remainingFM.parameters())  + list(imageGan.G.parameters()) , lr=0.0002, betas=(0.5, 0.999))
optimizer_discriminator = torch.optim.Adam( list(imageGan.D1.parameters()) + list(imageGan.D2.parameters()) + list(imageGan.D3.parameters()) , lr=0.0002, betas=(0.5, 0.999))

l1 = L1()
bce = BCE()
perceptual = Perceptual()

import cv2

transTensor = transforms.Compose([
        transforms.ToTensor()   # converting to grayscale
    ])

running_loss = {
            'loss_G' : 0,
            'loss_D' : 0
        }


import torch
import matplotlib.pyplot as plt
# torch.cpu.empty_cache() 
# torch.cpu.set_per_process_memory_fraction(0.5) 

count=0
print("here")

epochs=3
# mlflow.set_tracking_uri("http://localhost:5000")
# mlflow.set_experiment("MlopsProject")


# with mlflow.start_run(run_name="run1", description="MlopsProject") as run:
for i in range(30001):
    if i in l_eyeSegments:

        realPhoto = cv2.imread(realImages[i])
        realPhoto = cv2.resize(realPhoto, (512,512), interpolation=cv2.INTER_AREA)
        realPhoto=transTensor(realPhoto).unsqueeze(0)

        l_eyeSeg = Image.open(l_eyeSegments[i])
        l_eyeTen = transformFun(l_eyeSeg)
        l_eyeCEInput=l_eyeTen.unsqueeze(0)
        l_eyeCEOuput = l_eyeEncoder(l_eyeCEInput).to("cpu")
        l_eyeFMOuput = l_eyeFM(l_eyeCEOuput)
        # print(l_eyeFMOuput.shape)


        r_eyeSeg = Image.open(r_eyeSegments[i])
        r_eyeTen = transformFun(r_eyeSeg)
        r_eyeCEInput=r_eyeTen.unsqueeze(0)
        r_eyeCEOuput = r_eyeEncoder(r_eyeCEInput).to("cpu")
        r_eyeFMOuput = r_eyeFM(r_eyeCEOuput)
        # print(r_eyeFMOuput.shape)


        noseSeg = Image.open(noseSegments[i])
        noseTen = transformFun(noseSeg)
        noseCEInput=noseTen.unsqueeze(0)
        noseCEOuput = noseEncoder(noseCEInput).to("cpu")
        noseFMOuput = noseFM(noseCEOuput)
        # print(noseFMOuput.shape)


        mouthSeg = Image.open(mouthSegments[i])
        mouthTen = transformFun(mouthSeg)
        mouthCEInput=mouthTen.unsqueeze(0)
        mouthCEOuput = mouthEncoder(mouthCEInput).to("cpu")
        mouthFMOuput = mouthFM(mouthCEOuput)
        # print(mouthFMOuput.shape)

        remainingSeg = Image.open(remainingSegments[i])
        remainingTen = transformFun(remainingSeg)
        remainingCEInput=remainingTen.unsqueeze(0)
        remainingCEOuput = remainingEncoder(remainingCEInput).to("cpu")
        remainingFMOuput = remainingFM(remainingCEOuput)
        # print(remainingFMOuput.shape)


        allFMOutputs = [l_eyeFMOuput,r_eyeFMOuput,noseFMOuput,mouthFMOuput,remainingFMOuput]

        mergedOuput = segmentImage(anotImages,hedAfterSimo,i,allFMOutputs)

        GanOutput = imageGan.generate(mergedOuput)

        # import copy
        # if count%10==0:
        #     image = cv2.cvtColor(GanOutput.clone().cpu().squeeze(0).permute(1,2,0).detach().numpy(),cv2.COLOR_RGB2BGR)


        #     plt.imshow(image)
        #     plt.show()


        optimizer_generator.zero_grad()
        loss_G_L1 = l1.compute(GanOutput.to("cpu"), realPhoto.to("cpu"))
        loss_perceptual = perceptual.compute(GanOutput.to("cpu"), realPhoto.to("cpu"))
        patches = imageGan.discriminate(mergedOuput.to("cpu"), GanOutput.to("cpu"))
        loss_G_BCE = torch.tensor([bce.compute(patch, torch.full(patch.shape, label_real, dtype=torch.float, requires_grad=True).to("cpu")) for patch in patches], dtype=torch.float, requires_grad=True).sum()
        loss_G = loss_perceptual + 10 * loss_G_L1 + loss_G_BCE
        loss_G.backward()
        optimizer_generator.step()

        optimizer_discriminator.zero_grad()
        patches = imageGan.discriminate(mergedOuput.detach().to("cpu"), GanOutput.detach().to("cpu"))
        loss_D_fake = torch.tensor([bce.compute(patch, torch.full(patch.shape, label_fake, dtype=torch.float, requires_grad=True).to("cpu")) for patch in patches], dtype=torch.float, requires_grad=True).sum()
        patches = imageGan.discriminate(mergedOuput.detach().to("cpu"), realPhoto.detach().to("cpu"))
        loss_D_real = torch.tensor([bce.compute(patch, torch.full(patch.shape, label_real, dtype=torch.float, requires_grad=True).to("cpu")) for patch in patches], dtype=torch.float, requires_grad=True).sum()
        loss_D = loss_D_fake + loss_D_real
        loss_D.backward()
        optimizer_discriminator.step()

        count+=1
        iteration_loss = {
            'loss_G_it' : loss_G.item()/10,
            'loss_D_it' : loss_D.item()/10
        }
        print(i,count,iteration_loss)

        # mlflow.log_metric("loss_G_it", loss_G.item() / 10, step=i)
        # mlflow.log_metric("loss_D_it", loss_D.item() / 10, step=i)


torch.save(imageGan.G.state_dict(), rf"generator-{epochs}-{i}.pth")
torch.save(imageGan.D1.state_dict(), rf"D1-{epochs}-{i}.pth")
torch.save(imageGan.D2.state_dict(), rf"D2-{epochs}-{i}.pth")
torch.save(imageGan.D3.state_dict(), rf"D3-{epochs}-{i}.pth")

torch.save(l_eyeFM.state_dict(), rf"l_eyeFM-{epochs}-{i}.pth")
torch.save(r_eyeFM.state_dict(), rf"r_eyeFM-{epochs}-{i}.pth")
torch.save(noseFM.state_dict(), rf"noseFM-{epochs}-{i}.pth")
torch.save(mouthFM.state_dict(), rf"mouthFM-{epochs}-{i}.pth")
torch.save(remainingFM.state_dict(), rf"remainingFM-{epochs}-{i}.pth")



    
    # mlflow.pytorch.save_model(imageGan.G, "generator")
    # mlflow.pytorch.save_model(imageGan.D1, "D1")
    # mlflow.pytorch.save_model(imageGan.D2, "D2")
    # mlflow.pytorch.save_model(imageGan.D3, "D3")
    # mlflow.pytorch.save_model(l_eyeFM, "l_eyeFM")
    # mlflow.pytorch.save_model(r_eyeFM, "r_eyeFM")
    # mlflow.pytorch.save_model(noseFM, "noseFM")
    # mlflow.pytorch.save_model(mouthFM, "mouthFM")
    # mlflow.pytorch.save_model(remainingFM, "remainingFM")

