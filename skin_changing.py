from skinDetection import change_skin
import cv2
import numpy as np

def changeSkinColor(white , brown , darkbrown):

	input_image_path = "static\images\out.png"

	color = []
	if darkbrown:
		color = [115,71,55] # Dark brown
	elif brown:
		color = [186,130,73] # light brown
	elif white:
		color = [240,183,125] #white

	result=change_skin(input_image_path,[color[0],color[1],color[2]])
	with open("static\images\out.png",'wb') as resultFile:
		resultFile.write(result)

	# cv2.imshow(result,"hello")
	# cv2.waitKey(0)