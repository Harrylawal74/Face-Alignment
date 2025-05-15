import numpy as np
from sklearn.linear_model import Ridge
import cv2

'''
# Load the data using np.load
data = np.load('face_alignment_training_images.npz', allow_pickle=True)

# Extract the images
images = data['images']
# and the data points
landmarks = data['points']

print(images.shape, landmarks.shape)

testData = np.load('face_alignment_test_images.npz', allow_pickle=True)
testImages = testData['images']
print(testImages.shape)
plt.show()

def visualise_pts(image, landmarks):
  import matplotlib.pyplot as plt
  plt.imshow(image)
  plt.plot(ptlandmarkss[:, 0], landmarks[:, 1], '+r')
  plt.show()

for i in range(3):
  visualise_pts(images[i, ...], landmarks[i, ...])
'''
def loadTrainingData(images, landmarks):
    trainingData = []
    referenceShape = None
    for i in range(len(images)):
        image = images[i]
        referenceShape = landmarks[i]
        trainingData.append((image, referenceShape))
    return trainingData, referenceShape


# Estimating an initial shape from bounding box

def estimageInitialShape(bBox, referenceShape):
    x, y, w, h = bBox
    ref = np.array(referenceShape)

    # Normalising the reference shape then scaling it to the bounding box
    refNormalised = (ref - ref.min(0)) / (ref.max(0) - ref.min(0))
    scaled = refNormalised * [w, h] + [x, y]
    return scaled

# Using bounding box region. Extracting SIFT features from image
def extractSIFTFeatures(image, points):
    sift = cv2.SIFT_create()
    descriptors = []
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    for (x, y) in points:
        keyPoint = cv2.KeyPoint(float(x), float(y), 16)
        _, interestRegion = sift.compute(gray, [keyPoint])
        if interestRegion is not None:
            descriptors.append(interestRegion[0])
        else:
            descriptors.append(np.zeros(128))
    return np.concatenate(descriptors)



# Training the regressor model using the training data
def trainingRegressorModel(trainingData, referenceShape):
    xTrain, yTrain = [], []
    for image, realShape in trainingData:
        face = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face.detectMultiScale(gray, 1.3, 5)
        if len(faces) == 0:
            continue
        initialShape = estimageInitialShape(faces[0], referenceShape)
        features = extractSIFTFeatures(image, initialShape)
        delta = (realShape - initialShape).flatten()
        xTrain.append(features)
        yTrain.append(delta)
    model = Ridge(alpha=1.0)
    model.fit(xTrain, yTrain)
    return model, referenceShape






def predictLandmarks(image, model, referenceShape):
    face = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face.detectMultiScale(gray, 1.3, 5)
    if len(faces) == 0:
        return None
    initialShape = estimageInitialShape(faces[0], referenceShape)
    features = extractSIFTFeatures(image, initialShape)
    delta = model.predict([features])[0]
    predicted = initialShape + delta.reshape(-1, 2)
    return predicted



# Landmarks are dran onto the image in blue
def drawLandmarks(image, points, color=(0, 255, 0)):
    for (x, y) in points:
        cv2.circle(image, (int(x), int(y)), 3, color, -1)
    return image
