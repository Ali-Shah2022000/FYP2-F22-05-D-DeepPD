import torch
from torch import nn
from PIL import Image
from torchvision import datasets, transforms
from autoEncoder import autoEncoder


def predictImage(ten):
   criterion = nn.MSELoss()
   model = autoEncoder("eye")
   model.load_state_dict(torch.load('Models/l_eye.pt'))
   model.eval()
   # ten=torch.unsqueeze(trainLoader.dataset[80][0],0)
   output = model.forward(ten)
   loss = criterion(output, ten)
   print(loss)


transformFun = transforms.Compose([
        transforms.Grayscale(),     # converting to grayscale, so that we can get 1 channel
        transforms.ToTensor()   # converting to grayscale
    ])
img = Image.open(r'E:\FYP final\iteration1Results\NewResizedImage\resultsSegHed\l_eye\3359.jpg')
ten = transformFun(img).unsqueeze(0)
predictImage(ten)
