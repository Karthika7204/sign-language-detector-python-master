import os
import cv2

#set up the data folder
DATA_DIR = './data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

#collected image size
dataset_size = 5

# start the webcam
cap = cv2.VideoCapture(0)

# checking if the camera is working properly
if not cap.isOpened():
    print("❌ Webcam not detected")
    exit()
else:
    print("✅ Webcam is working")

# Loop for a each Class
for j in range(23,24):

    # Create Folder for That Class (E.g : Creates ./data/20 for saving class 20 images.)
    class_dir = os.path.join(DATA_DIR, str(j))
    if not os.path.exists(class_dir):
        os.makedirs(class_dir)

    print('Collecting data for class {}'.format(j))

    # Prompt user before starting collection & Wait Until User is Ready to press the q
    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            print("❌ Failed to read from webcam. Skipping this frame...")
            continue
        cv2.putText(frame, 'Ready? Press "Q" ! :)', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)
        cv2.imshow('frame', frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    # Collect Images size mentioned in the datasize 
    counter = 0
    while counter < dataset_size:
        ret, frame = cap.read()
        if not ret or frame is None:
            print("❌ Skipping invalid frame...")
            continue
        cv2.imshow('frame', frame)
        cv2.waitKey(1)
        img_path = os.path.join(class_dir, '{}.jpg'.format(counter))
        cv2.imwrite(img_path, frame)
        counter += 1

cap.release()
cv2.destroyAllWindows()
