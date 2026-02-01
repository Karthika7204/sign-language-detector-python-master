import pickle
import cv2
import mediapipe as mp
import numpy as np
import pygame
import pyttsx3
import time
import platform
import asyncio

# Initialize pygame mixer (optional, only if needed for other audio)
pygame.mixer.init()

model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

labels_dict = {0: 'A', 1: 'B', 2: 'C', 13:'D', 4: 'E',11: 'F', 3: 'H', 7: 'I', 12: 'K', 5: 'L',9: 'M', 10: 'N', 6: 'O',
               8: 'S',14:'T',15:'R',16:'V',17:'W',18:'Y',19:'U'}

current_word = []  # Store letters of the current word
sentence = []  # Store words of the sentence
last_letter = None  # Track the last detected letter to avoid repeats
last_detection_time = 0  # Track time of last detection
detection_delay = 1.0  # 1-second delay to stabilize detection
current_letter = None  # Track the currently detected letter
letter_start_time = 0  # Track when the current letter was first detected

def speak_sentence(sentence, current_word):
    """Generate and play the full sentence using pyttsx3."""
    sentence_to_speak = sentence + ([ ''.join(current_word) ] if current_word else [])
    sentence_to_speak = [word.lower() for word in sentence_to_speak]
    full_sentence = ' '.join(sentence_to_speak)
    
    if full_sentence:
        engine = pyttsx3.init()
        engine.say(full_sentence)
        engine.runAndWait()

async def main():
    global last_detection_time, last_letter, current_word, sentence, current_letter, letter_start_time

    while True:
        data_aux = []
        x_ = []
        y_ = []

        ret, frame = cap.read()

        if not ret or frame is None:
            continue

        H, W, _ = frame.shape

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = hands.process(frame_rgb)
        predicted_character = None  # Current frame's predicted character

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame,  # image to draw
                    hand_landmarks,  # model output
                    mp_hands.HAND_CONNECTIONS,  # hand connections
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())

            for hand_landmarks in results.multi_hand_landmarks:
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y

                    x_.append(x)
                    y_.append(y)

                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x - min(x_))
                    data_aux.append(y - min(y_))

            x1 = int(min(x_) * W) - 10
            y1 = int(min(y_) * H) - 10

            x2 = int(max(x_) * W) - 10
            y2 = int(max(y_) * H) - 10

            prediction = model.predict([np.asarray(data_aux)])
            predicted_character = labels_dict[int(prediction[0])]

            # Track the current letter and its duration
            current_time = time.time()
            if predicted_character != current_letter:
                current_letter = predicted_character
                letter_start_time = current_time

            # Only append the letter if it has been stable for 1 second and different from last added
            if (current_time - letter_start_time) >= detection_delay and predicted_character != last_letter:
                current_word.append(predicted_character)
                last_letter = predicted_character
                last_detection_time = current_time
                current_letter = None  # Reset to allow new letter detection

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
            cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
                        cv2.LINE_AA)

        # Display current word and sentence on the frame
        current_word_str = ''.join(current_word)
        sentence_display = sentence + ([current_word_str] if current_word_str else [])
        sentence_str = ' '.join(sentence_display)
        cv2.putText(frame, f'Word: {current_word_str}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, f'Sentence: {sentence_str}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.imshow('frame', frame)
        key = cv2.waitKey(1) & 0xFF

        # Press space bar to finish current word and start a new one
        if key == ord(' '):
            if current_word:  # Only append if there's a word
                sentence.append(''.join(current_word))
                current_word = []  # Reset current word
                last_letter = None  # Reset last letter to allow new detection
                last_detection_time = 0  # Reset detection time
                current_letter = None  # Reset current letter
                letter_start_time = 0  # Reset letter start time

        # Press 's' to speak the entire sentence
        if key == ord('s'):
            speak_sentence(sentence, current_word)

        # Press 'q' to quit
        if key == ord('q'):
            break

        await asyncio.sleep(1.0 / 60)  # Control frame rate

if platform.system() == "Emscripten":
    asyncio.ensure_future(main())
else:
    if __name__ == "__main__":
        asyncio.run(main())

cap.release()
cv2.destroyAllWindows()