# Importing Libraries
import os
import pickle
import mediapipe as mp
import cv2

# Setting Up MediaPipe Hands model
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# This is the folder where your first code saved hand gesture images.
DATA_DIR = './data'


data = [] # will store landmark coordinates for each image.
labels = [] # will store the class (folder name) of each image.


# Loop Through All Images present in different directory
for dir_ in os.listdir(DATA_DIR):
    for img_path in os.listdir(os.path.join(DATA_DIR, dir_)):
        data_aux = []

        # Variables for Landmark Processing
        x_ = []
        y_ = []

        # Read and Convert Image from (OpenCV default) to RGB (needed by MediaPipe).
        img = cv2.imread(os.path.join(DATA_DIR, dir_, img_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Passes the image into the MediaPipe Hands model Returns a list of 21 landmark points if a hand is found.
        results = hands.process(img_rgb) 
        if results.multi_hand_landmarks:

            # Extract Landmark Coordinates
            for hand_landmarks in results.multi_hand_landmarks:
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y

                    x_.append(x)
                    y_.append(y)

                # Normalize the Coordinates
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x - min(x_))
                    data_aux.append(y - min(y_))

            # save Data and Label
            data.append(data_aux)
            labels.append(dir_)

# Save everything in the data file 
f = open('data.pickle', 'wb')
pickle.dump({'data': data, 'labels': labels}, f)
f.close()
