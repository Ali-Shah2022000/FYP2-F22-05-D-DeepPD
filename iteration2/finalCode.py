# importing the necessary libraies and python files
from segmenting import *
from trainLoader import *
from autoEncoder import  *
import os

isExist = os.path.exists("hedSegments")   # checking if folder to save segments exist
if not isExist:   # if folder it exists , create directories and sub directories
   os.makedirs("hedSegments")
   os.makedirs("hedSegments/l_eye/l_eye")
   os.makedirs("hedSegments/r_eye/r_eye")
   os.makedirs("hedSegments/nose/nose")
   os.makedirs("hedSegments/mouth/mouth")
   os.makedirs("hedSegments/remaining/remaining")

isExistModel = os.path.exists("Models")

if not isExistModel:
   os.makedirs("Models")


segmentAllImages("AnotImages","Images","hedSegments")  # Segmenting images and saving the segments in respective folders

# comment above code (not necessary) when before running the following code
trainLoader = getTrainLoader("hedSegments/l_eye")

# second parameter is part,  [eye,nose,mouth,face]
EncodeAndDecode(trainLoader,"eye","l_eye")




