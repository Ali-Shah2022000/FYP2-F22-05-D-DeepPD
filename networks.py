import torch
from torch import nn


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


