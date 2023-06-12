import sys
import os
import cv2
import numpy as np
from skimage import io
from imutils import face_utils

# This function is meant to give coordinates of both eyes and mouth so skin detection can work better.
def getEyesMouth(img):

    # Load the facial landmarks predictor.
    predictor_path = "shape_predictor_68_face_landmarks.dat"
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)

    # Convert the image to grayscale.
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale image.
    rects = detector(gray, 0)

    # Make lists for storing the coordinates.
    leyeList=[]
    reyeList=[]
    mouthList=[]

    # Apply a for loop for more than one face in the image
    for rect in rects:

        # Get the facial landmarks for the face region.
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        # The various indices here are the predefined indices where the coordinates had been stored by the predictor.
        for i in range(36,42):
            leyeList.append(shape[i])
        for i in range(42,48):
            reyeList.append(shape[i])
        for i in range(48,60):
            mouthList.append(shape[i])

    # Return the lists for left eye, right eye and mouth as a tuple of lists.
    return (leyeList, reyeList, mouthList)
