# importing all the necessary libraries
from flask import Flask, render_template, jsonify, request, g,redirect
import base64
import matplotlib.pyplot as plt
import cv2
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import glob
import joblib
from PIL import Image
import glob
import copy
import torch
import torch
import torchvision.transforms as transforms
from PIL import Image
import networks
import ImageSynthisis as IS
import MergeImage as segImages
import utilFunctions as utils
import threading
from skin_changing import changeSkinColor

#==============================================================


transformFun = transforms.Compose([
        transforms.Grayscale(),     # converting to grayscale, so that we can get 1 channel
        transforms.ToTensor()   # converting to grayscale
    ])

# =========================================================================================
# variables that will be used to generate images and store Models
male_labels = np.array(glob.glob(r"static\sketchShadowKnnSmall\male\*.jpg"))
female_labels = np.array(glob.glob(r"static\sketchShadowKnnSmall\female\*.jpg"))
train_labels = male_labels	# to give some value initially
image_no = 0
anotImages=None
hedAfterSimo = None
function_completed = False
alg = cv2.KAZE_create()     # cv2 descriptor to find the features 
knnMale = joblib.load("maleShadowKNN.joblib")       # loading the Knn model for male shadow sketches
knnFemale = joblib.load("femaleShadowKNN.joblib")   # loading the Knn model for female shadow sketches
l_eyeEncoder =None
r_eyeEncoder = None
noseEncoder = None
mouthEncoder = None
remainingEncoder = None
l_eyeFM =None
r_eyeFM =None
noseFM =None
mouthFM =None
remainingFM =None
imageGan = None
knn=knnMale
# =========================================================================================


# =========================================================================================
def GenerateImage(anotImages,hedAfterSimo,i):       # function to generate the image from sketch
    print("in segmentation" , i)
    img = plt.imread("image.png",cv2.IMREAD_UNCHANGED)   # reading the sketch user drawn 
    alpha = img[:, :, 3]        # get the alpha layer of the sketch
    otherImage = np.zeros_like(img[:, :, :3])    # make an empty image 
    otherImage[alpha == 0] = [0, 0, 0]      # make the black part black in new image
    otherImage[alpha != 0] = [255, 255, 255]     # make the white part white in new image
    otherImage=cv2.cvtColor(otherImage, cv2.COLOR_BGR2GRAY)     # converting the image to grayscale
    print(otherImage.shape)

    fullImage = cv2.imread(hedAfterSimo[i])      # read the full image
    remainingFace = copy.deepcopy(otherImage)   # making a deep copy of the full image so that it does not change
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
            blackAndWhiteImg = cv2.imread(j)        #reading the image
            segmentedImages[2] = cv2.bitwise_or(segmentedImages[2], blackAndWhiteImg)

        elif attribute == "mouth" or attribute ==" l_lip"  or attribute == "u_lip":  #for mouth
            blackAndWhiteImg = cv2.imread(j)  #reading the image
            segmentedImages[3] = cv2.bitwise_or(segmentedImages[3], blackAndWhiteImg)

    allFMOutputs = [None,None,None,None,None]       # creating a list to save all the FM outputs so that they can be merged later
    for k in range(4):
        segmentedImages[k] = cv2.resize(segmentedImages[k], dimColoredImage, interpolation=cv2.INTER_AREA)      # resizing the images to 512
        gray = cv2.cvtColor(segmentedImages[k], cv2.COLOR_BGR2GRAY)         # converting the image to gray scale
        contours, _ = cv2.findContours(gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)        # get the contours of the iamge
        x, y, w, h = cv2.boundingRect(contours[0])      # getting the bounding box cordinates of the contour

        if k==0 or k ==1 :    # eyes   0 =left   1==right     (from our prespective)

            if k==0:        # left eye
                crop_img = otherImage[y - 50:y + h + 30, x - 70:x + w + 30]      # cropping the image (manually set)
                remainingFace[y - 50:y + h + 30, x - 70:x + w + 30] = 0                # setting the box from the boudning box to 255 , so that we can remove that segment
                crop_img=cv2.resize(crop_img, (128,128), interpolation=cv2.INTER_AREA)      # resizing
                crop_img = Image.fromarray(crop_img)
                l_eyeTen = transformFun(crop_img)       # converting to grayscale and tensor
                l_eyeCEInput=l_eyeTen.unsqueeze(0)      # removing the batch number 
                l_eyeCEOuput = l_eyeEncoder(l_eyeCEInput)  # passing to encoder
                l_eyeFMOuput = l_eyeFM(l_eyeCEOuput)    # passing to feature mapping 
                allFMOutputs[0]=l_eyeFMOuput
                print(l_eyeFMOuput.shape)

            else:
                crop_img = otherImage[y - 50:y + h + 30, x - 30:x + w + 70]           # cropping the image (manually set)
                remainingFace[y - 50:y + h + 30, x - 30:x + w + 70] = 0           # setting the box from the boudning box to 255 , so that we can remove that segment
                crop_img=cv2.resize(crop_img, (128,128), interpolation=cv2.INTER_AREA)
                crop_img = Image.fromarray(crop_img)
                r_eyeTen = transformFun(crop_img)      # converting to grayscale and tensor
                r_eyeCEInput=r_eyeTen.unsqueeze(0)    # removing the batch number 
                r_eyeCEOuput = r_eyeEncoder(r_eyeCEInput)   # passing to encoder
                r_eyeFMOuput = r_eyeFM(r_eyeCEOuput)    # passing to feature mapping 
                allFMOutputs[1]=r_eyeFMOuput
                print(r_eyeFMOuput.shape)



                # cv2.imwrite("r_eye.jpg",crop_img,)  #saving the image#saving the image

        if k==2:   # nose
            crop_img = otherImage[y - 5:y + h + 35, x - 20:x + w + 20]            # cropping the image (manually set)
            remainingFace[y - 5:y + h + 35, x - 20:x + w + 20] = 0         # setting the box from the boudning box to 255 , so that we can remove that segment
            crop_img = cv2.resize(crop_img, (160, 160), interpolation=cv2.INTER_AREA)
            crop_img = Image.fromarray(crop_img)
            noseTen = transformFun(crop_img)      # converting to grayscale and tensor
            noseCEInput=noseTen.unsqueeze(0)    # removing the batch number 
            noseCEOuput = noseEncoder(noseCEInput)  # passing to encoder
            noseFMOuput = noseFM(noseCEOuput)    # passing to feature mapping 
            allFMOutputs[2]=noseFMOuput
            print(noseFMOuput.shape)
            # cv2.imwrite(rf"nose.jpg", crop_img, )  #saving the image


        if k==3: # mouth
            crop_img = otherImage[y - 40:y + h + 70, x - 25:x + w + 25]           # cropping the image (manually set)
            remainingFace[y - 40:y + h + 70, x - 25:x + w + 25] = 0         # setting the box from the boudning box to 255 , so that we can remove that segment
            crop_img = cv2.resize(crop_img, (192, 192), interpolation=cv2.INTER_AREA)
            crop_img = Image.fromarray(crop_img)
            mouthTen = transformFun(crop_img)      # converting to grayscale and tensor
            mouthCEInput=mouthTen.unsqueeze(0)    # removing the batch number 
            mouthCEOuput = mouthEncoder(mouthCEInput)  # passing to encoder
            mouthFMOuput = mouthFM(mouthCEOuput)    # passing to feature mapping 
            allFMOutputs[3]=mouthFMOuput
            print(mouthFMOuput.shape)
            # cv2.imwrite(rf"mouth.jpg", crop_img, )  #saving the image

    remainingFace = cv2.resize(remainingFace, (512, 512), interpolation=cv2.INTER_AREA)
    crop_img = Image.fromarray(remainingFace)
    remainingTen = transformFun(crop_img)      # converting to grayscale and tensor
    remainingCEInput=remainingTen.unsqueeze(0)    # removing the batch number 
    remainingCEOuput = remainingEncoder(remainingCEInput)  # passing to encoder
    remainingFMOuput = remainingFM(remainingCEOuput)    # passing to feature mapping 
    allFMOutputs[4]=remainingFMOuput

    
    mergedOuput = segImages.mergeImage(anotImages,hedAfterSimo,i,allFMOutputs)      # merging the image
    GanOutput = imageGan.generate(mergedOuput)      # calling the image synthesis module
    numpy_image = GanOutput.squeeze().cpu().detach().numpy()      # converting to numpy  

    # Convert from RGB to BGR
    numpy_image = numpy_image[..., ::-1]        

    # Rescale the values to [0, 255] and convert to uint8
    numpy_image = (numpy_image - np.min(numpy_image)) / (np.max(numpy_image) - np.min(numpy_image)) * 255
    numpy_image = numpy_image.astype(np.uint8)

    # Convert numpy array to PIL Image
    pil_image = Image.fromarray(np.transpose(numpy_image, (1, 2, 0)))

    pil_image.save('image.png')     #saving the image

        
    cv2_image = cv2.imread('image.png')     #reading the image
    rgb_image = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)  
    cv2.imwrite('static\images\out.png', rgb_image)

    # import requests     
    # response = requests.post(           # calling the api to deblur the image
    #     'https://www.cutout.pro/api/v1/matting?mattingType=18',
    #     files={'file': open('static\images\out.png', 'rb')},
    #     headers={'APIKEY': '7ec9b38ba5924c308314c88b5a15bb68'},
    # )
    # with open('static\images\out.png', 'wb') as out:        
    #     out.write(response.content)
    
    print("generatedImage")
    global function_completed
    function_completed = True

        # return merged
# =========================================================================================
# Extract features from the images
def extract_features(image, alg, vector_size=32):
    try:
        # Detect keypoints in the image
        kps = alg.detect(image)
        # Sort keypoints by response and keep top vector_size keypoints
        kps = sorted(kps, key=lambda x: -x.response)[:vector_size]
        # Compute keypoints descriptors
        kps, dsc = alg.compute(image, kps)
        if len(kps) != 0:
            dsc = dsc.flatten()
            needed_size = (vector_size * 64)
            if dsc.size < needed_size:
                dsc = np.concatenate([dsc, np.zeros(needed_size - dsc.size)])
            dsc = dsc[:2048]
        else:
            return np.zeros(vector_size * 64)
    except cv2.error as e:
        print('Error:', e)
        return None
    # Sort descriptor indices in descending order
    idx = dsc.argsort()[-1:-411:-1]
    binarize_dsc = np.zeros(2048)
    # Set corresponding indices to 1 in binarize_dsc
    binarize_dsc[idx[:]] = 1
    return binarize_dsc

# =========================================================================================

# =========================================================================================
# Fit the knn model using the extracted features
def fit_knn(train_data, train_labels):
    # Create an instance of KNeighborsClassifier with cosine metric
    knn = KNeighborsClassifier(metric='cosine')
    # Fit the training data and labels to the KNN classifier
    knn.fit(train_data, train_labels)
    # Return the trained KNN classifier
    return knn
# =========================================================================================

# =========================================================================================
# Predict the closest neighbor using the knn model
def predict_neighbors(knn, test_data, k):
	distances, indices = knn.kneighbors(test_data, n_neighbors=20)
	neighbor_labels = train_labels[indices]
	return distances, neighbor_labels
# =========================================================================================

# =========================================================================================
def get_train_data(folder_path, alg):
    # Get all image file paths in the specified folder
    allImages = glob.glob(folder_path + "\*.jpg")
    train_data = []
    train_labels = []
    
    count = 0
    for i in allImages:
        # Read and resize the image
        img = cv2.imread(i)
        img = cv2.resize(img, (512, 512))
        # Extract features from the image using the provided algorithm
        features = extract_features(img, alg)
        # Append the features to train_data and the image path to train_labels
        train_data.append(features)
        train_labels.append(i)
        count += 1
        # Print the count to track progress
        print(count)

    return np.array(train_data), np.array(train_labels)

# =========================================================================================

# =========================================================================================
def start(image):
    global image_no

    # Resize the input image to (512, 512)
    img = cv2.resize(image, (512, 512))

    # Extract features from the resized image
    feature = extract_features(img, alg)

    # Predict the nearest neighbors using the trained KNN classifier
    distances, closestLabels = predict_neighbors(knn, np.array([feature]), k=10)

    # Initialize the necessary arrays
    final_image_created = np.zeros(shape=[512, 512])
    blur = np.zeros((512, 512), dtype=np.int32)
    most_matched_image = np.zeros((512, 512), dtype=np.int32)
    new_image_formed = np.zeros(shape=[512, 512])

    # Read the most matched image
    most_matched_image = cv2.imread(closestLabels[0][-1], cv2.IMREAD_GRAYSCALE)

    # Iterate through the top 10 closest labels
    for im in range(10):
        imgClose = cv2.imread(closestLabels[0][im], cv2.IMREAD_GRAYSCALE)
        mask = imgClose < 200
        new_image_formed[np.where(mask)] += ((2.31) ** (-1 * distances[0][im]))

    # Calculate the total match value
    total_match_value = np.sum(np.power(2.31, -1 * distances[0]))

    # Create the final image
    final_image_created = (1 - new_image_formed / total_match_value) * 255
    final_image_created = final_image_created.astype(np.uint8)

    # Apply blur to the final image
    blur = cv2.blur(final_image_created, (15, 15))

    # Blend the blurred image and the most matched image
    dst = cv2.addWeighted(blur, 0.8, most_matched_image, 0.2, 0, dtype=cv2.CV_32S)

    # Extract the closest matched sketch filename and image number
    print("Closest matched sketch:", closestLabels[0][0])  # This will show the whole path
    full_image_name = closestLabels[0][0].split('\\')[-1]  # This will get the image with extension
    print(full_image_name)
    image_no = int(full_image_name.split('.')[0])
    print(image_no)

    return dst

# =========================================================================================






# =========================================================================================
# =========================================================================================
# =========================================================================================
app = Flask(__name__)
@app.route('/')
def index():
    # Initialize and load models
    print("loading models now")

    global hedAfterSimo
    global anotImages
    global l_eyeEncoder
    global r_eyeEncoder
    global noseEncoder
    global mouthEncoder
    global remainingEncoder

    global l_eyeFM
    global r_eyeFM
    global noseFM
    global mouthFM
    global remainingFM

    global imageGan

    # Load hedAfterSimo model
    hedAfterSimo = utils.resizedHed(r'static/hedAfterSimo')

    # Load anotImages dictionary
    anotImages = utils.getAnotDict(r'static/AnotImages')

    # Load l_eyeEncoder model
    l_eyeEncoder = networks.define_part_encoder("eye")
    l_eyeEncoder.load_state_dict(torch.load(r"static/Models/l_eye.pt"))

    # Load r_eyeEncoder model
    r_eyeEncoder = networks.define_part_encoder("eye")
    r_eyeEncoder.load_state_dict(torch.load(r"static/Models/r_eye.pt"))

    # Load noseEncoder model
    noseEncoder = networks.define_part_encoder("nose")
    noseEncoder.load_state_dict(torch.load(r"static/Models/nose.pt"))

    # Load mouthEncoder model
    mouthEncoder = networks.define_part_encoder("mouth")
    mouthEncoder.load_state_dict(torch.load(r"static/Models/mouth.pt"))

    # Load remainingEncoder model
    remainingEncoder = networks.define_part_encoder("face")
    remainingEncoder.load_state_dict(torch.load(r"static/Models/remaining.pt"))

    # Load l_eyeFM model
    l_eyeFM = networks.featureMapping("eye")
    l_eyeFM.load_state_dict(torch.load(r"static/Models/l_eyeFM-5-30000.pth", map_location=torch.device("cpu")))

    # Load r_eyeFM model
    r_eyeFM = networks.featureMapping("eye")
    r_eyeFM.load_state_dict(torch.load(r"static/Models/r_eyeFM-5-30000.pth", map_location=torch.device('cpu')))

    # Load noseFM model
    noseFM = networks.featureMapping("nose")
    noseFM.load_state_dict(torch.load(r"static/Models/noseFM-5-30000.pth", map_location=torch.device('cpu')))

    # Load mouthFM model
    mouthFM = networks.featureMapping("mouth")
    mouthFM.load_state_dict(torch.load(r"static/Models/mouthFM-5-30000.pth", map_location=torch.device('cpu')))

    # Load remainingFM model
    remainingFM = networks.featureMapping("remaining")
    remainingFM.load_state_dict(torch.load(r"static/Models/remainingFM-5-30000.pth", map_location=torch.device('cpu')))

    # Load imageGan model
    imageGan = IS.GanModule()
    imageGan.G.load_state_dict(torch.load(r"static/Models/generator-5-30000.pth", map_location=torch.device('cpu')))

    print("loaded")

    # Redirect to 'initial' route
    return redirect('initial')


@app.route('/initial')
def initial():
    return render_template('index.html')

@app.route('/generateimage')
def gotogenerate():
    # Handle '/generateimage' route

    print("image no : ", image_no)

    global function_completed
    function_completed = False

    # Start a new thread to generate the image using the GenerateImage function
    thread = threading.Thread(target=GenerateImage, args=(anotImages, hedAfterSimo, image_no))
    thread.start()

    # Render the 'temp.html' template
    return render_template('temp.html')


@app.route('/getstatus')
def getstatus():
    # Handle '/getstatus' route

    global function_completed

    if function_completed:
        # If the image generation function has completed, return 'completed' status
        return jsonify({'status': 'completed'})
    else:
        # If the image generation function is still running, return 'running' status
        return jsonify({'status': 'running'})


@app.route('/final')
def finalimage():
    # Handle '/final' route

    print("generate now")

    # Render the 'generate.html' template
    return render_template('generate.html')


@app.route('/mainpage')
def conttomain():
    # Handle '/mainpage' route

    # Render the 'mainfile.html' template
    return render_template('mainfile.html')


train_labels = male_labels	# to give some value initially

@app.route("/change_label",methods=["POST"])
def change_label():
	global train_labels
	global knn

	data = request.get_json()
	# when user clicks on male or female button, a boolean is set and here is its value
	# true for male and false for female
	is_male = data['isMale']	
	if is_male==1:
		train_labels=male_labels
		knn=knnMale
	else:
		train_labels=female_labels
		knn=knnFemale

	# print(train_labels)
	return "Changed label"


@app.route("/update_shadow", methods=["POST"])
def update_shadow():
    # Retrieve the image data from the request
    dataURL = request.form.get("image")
    data = dataURL.split(',')[1]
    data = base64.b64decode(data)

    # Save the image to a file
    with open("image.png", "wb") as f:
        f.write(data)
    
    # Read the image using OpenCV and extract the alpha channel
    img = plt.imread("image.png", cv2.IMREAD_UNCHANGED)
    alpha = img[:, :, 3]

    # Create a black-and-white mask based on the alpha channel
    otherImage = np.zeros_like(img[:, :, :3])
    otherImage[alpha == 0] = [255, 255, 255]
    otherImage[alpha != 0] = [0, 0, 0]
    otherImage = cv2.cvtColor(otherImage, cv2.COLOR_BGR2GRAY)
 
    # Process the mask using the 'start' function to generate the shadow image
    shadowimage = start(otherImage)
    shadowimage = cv2.merge((shadowimage, shadowimage, shadowimage))

    # Convert the shadow image to base64 encoding
    _, buffer = cv2.imencode('.png', shadowimage)
    image_base64 = base64.b64encode(buffer).decode('utf-8')

    # Return the base64 encoded shadow image
    return image_base64

@app.route("/skins", methods=["POST"])
def skinchange():
    # Retrieve JSON data from the request
    skin_data = request.get_json()
    # Extract values from the JSON data
    is_white = skin_data['isWhite']
    is_brown = skin_data['isBrown']
    is_dbrown = skin_data['isDBrown']
    # Print the values of the skin colors
    print(is_white, is_brown, is_dbrown)
    # Call the function to change the skin color
    changeSkinColor(is_white, is_brown, is_dbrown)
    # Create a response dictionary
    response = {
        'status': 'success',
        'message': 'Skin change completed successfully'
    }    
    # Return the response as JSON
    return jsonify(response)


if __name__ == '__main__':
    app.run()