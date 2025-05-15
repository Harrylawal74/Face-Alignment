import numpy as np
import cv2
import matplotlib.pyplot as plt
from functions import * 

# Load training and test data
data = np.load('face_alignment_training_images.npz', allow_pickle=True)
images = data['images']
landmarks = data['points']
print(images.shape, landmarks.shape)

testData = np.load('face_alignment_test_images.npz', allow_pickle=True)
testImages = testData['images']
print(testImages.shape)
plt.show()

# Train the model using the training data
trainingData, referenceShape = loadTrainingData(images, landmarks)
model, referenceShape = trainingRegressorModel(trainingData, referenceShape)

# Printing the images with the predicted landmarks on them
# Testing the model on the test images
# If a face is detected then the predicted landmarks are drawn on the image
# If no face is detected then the image is shown but with no landmarks
for i in range(len(testImages)):
    test_image = testImages[i]
    predictedLandmarks = predictLandmarks(test_image, model, referenceShape)
    if predictedLandmarks is not None:
        output = drawLandmarks(test_image.copy(), predictedLandmarks)
        cv2.imshow("Predicted Landmarks", output)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        cv2.imshow("No landmarks", testImages[i])
        print("Did not detect any face")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
