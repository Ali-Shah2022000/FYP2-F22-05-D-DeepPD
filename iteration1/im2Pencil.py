import cv2
import glob
import os

def ConvertAllToSketch(InputImages,OutputFolder):
    mainImages = glob.glob(rf"{InputImages}/*.jpg")
    allImages = {}
    for image in mainImages:
        index = int(image[image.rindex("\\") + 1:-4])
        allImages[index] = image

    if (len(allImages) < 1):
        print("Error Removed Backgrounds Images Not found")
        exit(0)

    for i in allImages:
        img=cv2.imread(allImages[i])
        # img = cv2.resize(img, (0, 0), fx=0.6, fy=0.6)
        grey_img=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        grey_img=cv2.equalizeHist(grey_img)
        invert_img=cv2.bitwise_not(grey_img)
        blur_img=cv2.GaussianBlur(invert_img, (111,111),100)
        invblur_img=cv2.bitwise_not(blur_img)
        sketch_img=cv2.divide(grey_img,invblur_img, scale=256.0)

        where=allImages[i][allImages[i].rindex("\\")+1:allImages[i].rindex(".")]

        if os.path.exists(OutputFolder) == False:
            os.mkdir(OutputFolder)
        print("Converted to Sketch : ",i)
        cv2.imwrite(f"{OutputFolder}\{where}.jpg", sketch_img)

        # cv2.imshow(f"sketch image : {i}",sketch_img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
