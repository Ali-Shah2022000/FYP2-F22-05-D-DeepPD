import os
import torch
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image
from model import module
import glob


def infer_sketch_gan(inputFile,outputLoc,gpu,model):

    img = Image.open( inputFile ).convert('L')
    w, h  = img.size[0], img.size[1]
    pw    = 8-(w%8) if w%8!=0 else 0
    ph    = 8-(h%8) if h%8!=0 else 0
    immean, imstd, model_path, remote_model = get_default_args(model)
    gpu =gpu=="1" and torch.cuda.is_available()
    data  = ((transforms.ToTensor()(img)-immean)/imstd).unsqueeze(0)
    if pw!=0 or ph!=0:
        data = torch.nn.ReplicationPad2d( (0,pw,0,ph) )( data ).data
    data = data.float().cuda() if gpu else data.float()

    model = module.Net()
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path)
    else:
        checkpoint = torch.hub.load_state_dict_from_url(remote_model, progress=True)
    model.load_state_dict(checkpoint)
    model.eval()

    save_img_path = outputLoc+inputFile[inputFile.rindex("\\")+1:]
    with torch.no_grad():
        pred = model.cuda().forward( data ).float() if gpu else model.forward( data ).float()
        save_image(pred, save_img_path)


mean_std = {
    "gan": (0.9664114577640158, 0.0858381272736797, "../models/model_gan.pth", "https://github.com/aidreamwin/sketch_simplification_pytorch/releases/download/model/model_gan.pth"),
}

def get_default_args(model_name):
    return mean_std.get(model_name)


def SimiplifySketches(InputFolder,OutputFolder):

    mainImages = glob.glob(rf"{InputFolder}/*.jpg")
    allImages = {}
    for image in mainImages:
        index = int(image[image.rindex("\\") + 1:-4])
        allImages[index] = image

    if (len(allImages) < 1):
        print("Error Images Images Not found")
        exit(0)

    for i in range(0,30001):
        if i in allImages:
            if os.path.exists(OutputFolder) == False:
                os.mkdir(OutputFolder)
            infer_sketch_gan(inputFile=allImages[i],outputLoc=rf"{OutputFolder}\\",gpu="1",model="gan")
            print(f"Simplified : {i}")
