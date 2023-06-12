import glob
import networks
import torch
# =========================================================================================
def resizedHed(hedPath):    # function to get all the images
    mainImages = glob.glob(fr"{hedPath}\*.jpg")     # getting all images names and paths
    allImages = {}  # create a empty dictionary
    for image in mainImages:
        # print(image)
        index = int(image[image.rindex("\\") + 1:-4])   # get the index of the image (number)
        allImages[index] = image       # for the index, save the complete path of the image
    return allImages
# =========================================================================================

# =========================================================================================
def getAnotDict(anotPath):  # function to get all annotation images in a dictionary
    anotImages = glob.glob(f"{anotPath}\*.png")  # get all images
    anotImagesDict = {}     # create a empty dictionary
    for image in anotImages:
        oneImage = image[image.rindex("\\") + 1:]      # get the key (it will be a int number) +extension
        if int(oneImage[:oneImage.index("_")]) not in anotImagesDict:   # get the index and checking if it already exists
            anotImagesDict[int(oneImage[:oneImage.index("_")])] = []        # if not then create an empty list for that image
        anotImagesDict[int(oneImage[:oneImage.index("_")])].append(image)   # add anot image to that list with index
    return anotImagesDict
# =========================================================================================


def getSegments(Path):
    # Use glob to retrieve all file paths that match the given pattern
    segments = glob.glob(Path)

    # Create an empty dictionary to store the segments with their corresponding indices
    segmentsDict = {}

    # Iterate over each image path
    for image in segments:
        # Extract the index from the image path
        index = int(image[image.rindex("/") + 1:-4])

        # Add the image path to the dictionary with the index as the key
        segmentsDict[index] = image

    # Return the dictionary containing the segments
    return segmentsDict


# =========================================================================================


def load_EncoderModel(model_path, model_name):
    # Define the model using the specified model name
    model = networks.define_part_encoder(model_name)

    # Load the model weights from the provided model path
    model.load_state_dict(torch.load(model_path))

    # Return the loaded model
    return model
