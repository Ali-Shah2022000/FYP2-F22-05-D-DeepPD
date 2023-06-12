# importing the necessary images
import glob
import cv2
import numpy as np
import  copy

# variables to load the dictionaries of images and annotation images
anotImages=None
hedAfterSimo = None


def getAnotDict(anotPath):  # function to get all annotation images in a dictionary
    anotImages = glob.glob(f"{anotPath}\*.png")  # get all images
    anotImagesDict = {}     # create a empty dictionary
    for image in anotImages:
        oneImage = image[image.rindex("\\") + 1:]      # get the key (it will be a int number) +extension
        if int(oneImage[:oneImage.index("_")]) not in anotImagesDict:   # get the index and checking if it already exists
            anotImagesDict[int(oneImage[:oneImage.index("_")])] = []        # if not then create an empty list for that image
        anotImagesDict[int(oneImage[:oneImage.index("_")])].append(image)   # add anot image to that list with index
    return anotImagesDict

def resizedHed(hedPath):    # function to get all the images
    mainImages = glob.glob(rf"{hedPath}\*.jpg")     # getting all images names and paths
    allImages = {}  # create a empty dictionary
    for image in mainImages:
        index = int(image[image.rindex("\\") + 1:-4])   # get the index of the image (number)
        allImages[index] = image       # for the index, save the complete path of the image
    return allImages


def segmentImage(anotImages,hedAfterSimo,pathTosave):  # function to create segments
    for i in range(0,30001):    # run a loop for all images
        if i in hedAfterSimo:
            fullImage = cv2.imread(hedAfterSimo[i])      # read the full image
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


            for k in range(4):
                segmentedImages[k] = cv2.resize(segmentedImages[k], dimColoredImage, interpolation=cv2.INTER_AREA)      # resizing the images to 512
                gray = cv2.cvtColor(segmentedImages[k], cv2.COLOR_BGR2GRAY)         # converting the image to gray scale
                contours, _ = cv2.findContours(gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)        # get the contours of the iamge
                x, y, w, h = cv2.boundingRect(contours[0])      # getting the bounding box cordinates of the contour

                if k==0 or k ==1 :    # eyes   0 =left   1==right     (from our prespective)

                    if k==0:        # left eye
                        crop_img = fullImage[y - 50:y + h + 30, x - 70:x + w + 30]      # cropping the image (manually set)
                        remainingFace[y - 50:y + h + 30, x - 70:x + w + 30] = 255                # setting the box from the boudning box to 255 , so that we can remove that segment
                        crop_img=cv2.resize(crop_img, (128,128), interpolation=cv2.INTER_AREA)      # resizing
                        crop_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
                        crop_img = cv2.bitwise_not(crop_img)        # inverting the image
                        cv2.imwrite(rf"{pathTosave}\l_eye\l_eye\{i}.jpg",crop_img,)  #saving the image

                    else:
                        crop_img = fullImage[y - 50:y + h + 30, x - 30:x + w + 70]           # cropping the image (manually set)
                        remainingFace[y - 50:y + h + 30, x - 30:x + w + 70] = 255           # setting the box from the boudning box to 255 , so that we can remove that segment
                        crop_img=cv2.resize(crop_img, (128,128), interpolation=cv2.INTER_AREA)
                        crop_img = cv2.bitwise_not(crop_img)       # inverting the image
                        cv2.imwrite(rf"{pathTosave}\r_eye\r_eye\{i}.jpg", crop_img, )   #saving the image

                if k==2:   # nose
                    crop_img = fullImage[y - 5:y + h + 35, x - 20:x + w + 20]            # cropping the image (manually set)
                    remainingFace[y - 5:y + h + 35, x - 20:x + w + 20] = 255         # setting the box from the boudning box to 255 , so that we can remove that segment
                    crop_img = cv2.resize(crop_img, (160, 160), interpolation=cv2.INTER_AREA)
                    crop_img = cv2.bitwise_not(crop_img)       # inverting the image
                    cv2.imwrite(rf"{pathTosave}\nose\nose\{i}.jpg", crop_img, )  #saving the image


                if k==3: # mouth
                    crop_img = fullImage[y - 40:y + h + 70, x - 25:x + w + 25]           # cropping the image (manually set)
                    remainingFace[y - 40:y + h + 70, x - 25:x + w + 25] = 255         # setting the box from the boudning box to 255 , so that we can remove that segment
                    crop_img = cv2.resize(crop_img, (192, 192), interpolation=cv2.INTER_AREA)
                    crop_img = cv2.bitwise_not(crop_img)
                    cv2.imwrite(rf"{pathTosave}\mouth\mouth\{i}.jpg", crop_img, )  #saving the image

            remainingFace = cv2.resize(remainingFace, (512, 512), interpolation=cv2.INTER_AREA)

            remainingFace = cv2.bitwise_not(remainingFace)       # inverting the image
            cv2.imwrite(rf"{pathTosave}\remaining\remaining\{i}.jpg", remainingFace, )  #saving the image
            print(i, "Done")

            # break


def segmentAllImages(anotPath,hedPath,pathTosave):      # function to segment all the images
    anotImages = getAnotDict(anotPath)
    hedAfterSimo = resizedHed(hedPath)

    segmentImage(anotImages,hedAfterSimo,pathTosave)


# cv2.imshow("mainImage", ColoredImage)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

