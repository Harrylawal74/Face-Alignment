import cv2
import numpy as np
from sklearn.linear_model import Ridge    
import matplotlib.pyplot as plt



# Load the data using np.load
data = np.load('face_alignment_training_images.npz', allow_pickle=True)

# Extract the images
images = data['images']
# and the data points
landmarks = data['points']

print(images.shape, landmarks.shape)

testData = np.load('face_alignment_test_images.npz', allow_pickle=True)
testImages = testData['images']
plt.show()
'''
def visualise_pts(img, landmarks):
  import matplotlib.pyplot as plt
  plt.imshow(img)
  plt.plot(ptlandmarkss[:, 0], landmarks[:, 1], '+r')
  plt.show()

for i in range(3):
  visualise_pts(images[i, ...], landmarks[i, ...])
'''


def load_training_data(images, landmarks):
    training_data = []
    ref_shape = None
    for i in range(len(images)):
        img = images[i]
        ref_shape = landmarks[i]
        training_data.append((img, ref_shape))
    return training_data, ref_shape


#Estimate initial shape from face box 
def estimate_initial_shape(face_rect, ref_shape):
    x, y, w, h = face_rect
    ref = np.array(ref_shape)
    # Normalize ref to (0-1), then scale to box
    ref_norm = (ref - ref.min(0)) / (ref.max(0) - ref.min(0))
    scaled = ref_norm * [w, h] + [x, y]
    return scaled


def extract_sift_features(img, points):
    sift = cv2.SIFT_create()
    descriptors = []
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    for (x, y) in points:
        kp = cv2.KeyPoint(float(x), float(y), 16)  
        _, desc = sift.compute(gray, [kp])
        if desc is not None:
            descriptors.append(desc[0])
        else:
            descriptors.append(np.zeros(128))
    return np.concatenate(descriptors)


def train_regressor(training_data, ref_shape):
    x_train, y_train, = [], []
    for img, true_shape in training_data:
        face = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face.detectMultiScale(gray, 1.3, 5)

        if len(faces) == 0:
            continue

        initial_shape = estimate_initial_shape(faces[0], ref_shape)
        features = extract_sift_features(img, initial_shape)
        delta = (true_shape - initial_shape).flatten()
        x_train.append(features)
        y_train.append(delta)
    model = Ridge(alpha=1.0)
    model.fit(x_train, y_train)
    return model, ref_shape


def predict_landmarks(img, points, ref_shape):
    face = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face.detectMultiScale(gray, 1.3, 5)
    if len(faces) == 0:
        return None
    initial_shape = estimate_initial_shape(faces[0], ref_shape)
    features = extract_sift_features(img, initial_shape)
    delta = model.predict([features])[0]
    predicted = initial_shape + delta.reshape(-1, 2)
    return predicted


def draw_landmarks(img, points, color=(0, 255, 0)):
    for (x, y) in points:
        cv2.circle(img, (int(x), int(y)), 3, color, -1)
    return img


training_data, ref_shape = load_training_data(images, landmarks)

model, ref_shape = train_regressor(training_data, ref_shape)


for i in range(len(testImages)):
    test_image = testImages[i]
    pred_landmarks = predict_landmarks(test_image, model, ref_shape)
    if pred_landmarks is not None:
        output = draw_landmarks(test_image.copy(), pred_landmarks)
        cv2.imshow("Predicted Landmarks", output)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        cv2.imshow("No landmarks", testImages[i])
        print("No face detected")
        cv2.waitKey(0)
        cv2.destroyAllWindows()