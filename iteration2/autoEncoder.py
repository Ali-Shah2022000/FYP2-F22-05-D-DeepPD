import torch
from torch import nn


def define_part_encoder(model='eye', norm='instance', input_nc=1, latent_dim=512):
    norm_layer = nn.BatchNorm2d
    image_size = 512
    if 'eye' in model:
        image_size = 128
    elif 'mouth' in model:
        image_size = 192
    elif 'nose' in model:
        image_size = 160
    elif 'face' in model:
        image_size = 512

    net_encoder = EncoderGenerator_Res(norm_layer, image_size, input_nc,latent_dim)
    return net_encoder


class EncoderBlock(nn.Module):
    def __init__(self, channel_in, channel_out, kernel_size=7, padding=3, stride=4):
        super(EncoderBlock, self).__init__()
        self.conv = nn.Conv2d(channel_in, channel_out, kernel_size, padding=padding, stride=stride)
        self.bn = nn.BatchNorm2d(channel_out, momentum=0.9)
        self.relu = nn.LeakyReLU(1)

    def forward(self, ten, out=False, t=False):
        if out:
            ten = self.conv(ten)
            ten_out = ten
            ten = self.bn(ten)
            ten = self.relu(ten)
            return (ten, ten_out)
        else:
            ten = self.conv(ten)
            ten = self.bn(ten)
            ten = self.relu(ten)
            return ten


class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, activation=nn.ReLU(True), use_dropout=False):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, activation, use_dropout)

    def build_conv_block(self, dim, padding_type, norm_layer, activation, use_dropout):
        conv_block = []
        p = 0
        if (padding_type == 'reflect'):
            conv_block += [nn.ReflectionPad2d(1)]
        elif (padding_type == 'replicate'):
            conv_block += [nn.ReplicationPad2d(1)]
        elif (padding_type == 'zero'):
            p = 1
        conv_block += [nn.Conv2d(dim, dim, 3, padding=p), norm_layer(dim), activation]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if (padding_type == 'reflect'):
            conv_block += [nn.ReflectionPad2d(1)]
        elif (padding_type == 'replicate'):
            conv_block += [nn.ReplicationPad2d(1)]
        elif (padding_type == 'zero'):
            p = 1
        conv_block += [nn.Conv2d(dim, dim, 3, padding=p), norm_layer(dim)]
        return nn.Sequential(*conv_block)

    def forward(self, x):
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

        layers_list.append(
            EncoderBlock(channel_in=input_nc, channel_out=32, kernel_size=4, padding=1, stride=2))  # 176 176

        dim_size = 32
        for i in range(4):
            layers_list.append(
                ResnetBlock(dim_size, padding_type=padding_type, activation=activation, norm_layer=norm_layer))
            layers_list.append(
                EncoderBlock(channel_in=dim_size, channel_out=dim_size * 2, kernel_size=4, padding=1, stride=2))
            dim_size *= 2

        layers_list.append(ResnetBlock(512, padding_type=padding_type, activation=activation, norm_layer=norm_layer))

        self.conv = nn.Sequential(*layers_list)
        self.fc_mu = nn.Sequential(nn.Linear(in_features=longsize, out_features=latent_dim))  # ,

    def forward(self, ten):
        ten = self.conv(ten)
        ten = torch.reshape(ten, [ten.size()[0], -1])
        mu = self.fc_mu(ten)
        return mu


# ============================================================================================

def define_part_decoder(model='eye', norm='instance', output_nc=1, latent_dim=512):
    norm_layer = nn.BatchNorm2d

    image_size = 512
    if 'eye' in model:
        image_size = 128
    elif 'mouth' in model:
        image_size = 192
    elif 'nose' in model:
        image_size = 160
    elif 'face' in model:
        image_size = 512

    decoder = DecoderGenerator_image_Res(norm_layer, image_size, output_nc,latent_dim)

    return decoder


class DecoderGenerator_image_Res(nn.Module):
    def __init__(self, norm_layer, image_size, output_nc, latent_dim=512):
        super(DecoderGenerator_image_Res, self).__init__()
        latent_size = int(image_size / 32)
        self.latent_size = latent_size
        longsize = 512 * latent_size * latent_size

        activation = nn.ReLU()
        padding_type = 'reflect'
        norm_layer = nn.BatchNorm2d

        self.fc = nn.Sequential(nn.Linear(in_features=latent_dim, out_features=longsize))
        layers_list = []

        layers_list.append(
            ResnetBlock(512, padding_type=padding_type, activation=activation, norm_layer=norm_layer))  # 176 176

        dim_size = 256
        for i in range(4):
            layers_list.append(
                DecoderBlock(channel_in=dim_size * 2, channel_out=dim_size, kernel_size=4, padding=1, stride=2,
                             output_padding=0))  # latent*2
            layers_list.append(
                ResnetBlock(dim_size, padding_type=padding_type, activation=activation, norm_layer=norm_layer))
            dim_size = int(dim_size / 2)

        layers_list.append(DecoderBlock(channel_in=32, channel_out=32, kernel_size=4, padding=1, stride=2,output_padding=0))  # 352 352

        layers_list.append(ResnetBlock(32, padding_type=padding_type, activation=activation, norm_layer=norm_layer))  # 176 176

        layers_list.append(nn.ReflectionPad2d(2))
        layers_list.append(nn.Conv2d(32, output_nc, kernel_size=5, padding=0))

        self.conv = nn.Sequential(*layers_list)

    def forward(self, ten):
        ten = self.fc(ten)
        ten = torch.reshape(ten, (ten.size()[0], 512, self.latent_size, self.latent_size))
        ten = self.conv(ten)
        return ten


class DecoderBlock(nn.Module):

    def __init__(self, channel_in, channel_out, kernel_size=4, padding=1, stride=2, output_padding=0, norelu=False):
        super(DecoderBlock, self).__init__()
        layers_list = []
        layers_list.append(nn.ConvTranspose2d(channel_in, channel_out, kernel_size, padding=padding, stride=stride,
                                              output_padding=output_padding))
        layers_list.append(nn.BatchNorm2d(channel_out, momentum=0.9))
        if (norelu == False):
            layers_list.append(nn.LeakyReLU(1))
        self.conv = nn.Sequential(*layers_list)

    def forward(self, ten):
        ten = self.conv(ten)
        return ten


class autoEncoder(nn.Module):
    def __init__(self,part):
        super(autoEncoder, self).__init__()
        self.encodermodel = define_part_encoder(part)
        self.decoderModel = define_part_decoder(part)

    def forward(self, x):
        encoded = self.encodermodel(x)
        decoded = self.decoderModel(encoded)
        return decoded



def EncodeAndDecode(train_dataloader,part,model):
    modelNamePath = "Models/"+model +".pt"

    finalModel = autoEncoder(part)

    betas = (0.5, 0.999)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(finalModel.parameters(),
                                 lr=0.0002,
                                 betas=betas,
                                 weight_decay=1e-5)

    i=0
    for (img, _) in train_dataloader.dataset:

        img = torch.unsqueeze(img, 0)

        decoded = finalModel(img)

        loss = criterion(decoded, img)
        print(loss, i)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        i += 1

    torch.save(finalModel.state_dict(), modelNamePath)


