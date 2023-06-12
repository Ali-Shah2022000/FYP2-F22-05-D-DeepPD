import cv2
import numpy as np
import glob
import  os


def removeAllBackground(InputImages,InputAttr,OutputFolder):
    anotImages = glob.glob(f"{InputAttr}\*.png")
    anotImagesDict={}
    for image in anotImages:
        oneImage=image[image.rindex("\\") + 1:]
        if int(oneImage[:oneImage.index("_")]) not in anotImagesDict:
            anotImagesDict[int(oneImage[:oneImage.index("_")])]=[]
        anotImagesDict[int(oneImage[:oneImage.index("_")])].append(image)

    if (len(anotImagesDict)<1):
        print("Error Attributes Images Not found")
        exit(0)

    mainImages = glob.glob(rf"{InputImages}/*.jpg")
    allImages={}
    for image in mainImages:
        index=int(image[image.rindex("\\")+1:-4])
        allImages[index]=image

    if (len(allImages)<1):
        print("Error Images Images Not found")
        exit(0)

    for i in anotImagesDict:
        print("Removing background of : ",i)
        img=cv2.imread(anotImagesDict[i][0])
        img=np.zeros((img.shape[0],img.shape[1],3), np.uint8)

        for j in anotImagesDict[i]:
            attribute = j[j.rindex("_") + 1:j.rindex(".")]
            if attribute!="cloth":
                imgOther = cv2.imread(j)
                img = cv2.bitwise_or(img, imgOther)

        normalImage=cv2.imread(allImages[i])
        dim=(normalImage.shape[0],normalImage.shape[1])

        resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
        resized=cv2.bitwise_not(resized)

        normalImage = cv2.bitwise_or(normalImage, resized)
        # normalImage = cv2.resize(normalImage, (0, 0), fx=0.6, fy=0.6)

        if os.path.exists(OutputFolder) == False:
            os.mkdir(OutputFolder)
        cv2.imwrite(fr"{OutputFolder}\{i}.jpg", normalImage)

        # cv2.imshow(f"RemovedBackground : {i}",resized)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
