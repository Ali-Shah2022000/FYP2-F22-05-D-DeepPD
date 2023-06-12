# importing all the important libraries
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

def getTrainLoader(pathToDir):   # function to get the trainloader

    transformFun = transforms.Compose([
        transforms.Grayscale(),     # converting to grayscale, so that we can get 1 channel
        transforms.ToTensor()   # converting to grayscale
    ])

    train_data = datasets.ImageFolder(root=pathToDir,  # folder where images are saved
                                      transform=transformFun,  # transform function
                                      target_transform=None)


    train_dataloader = DataLoader(dataset=train_data,      # makeing the train_Dataloader
                                  batch_size=16,
                                  num_workers=1,
                                  shuffle=False)

    return train_dataloader     # returning the data loader

