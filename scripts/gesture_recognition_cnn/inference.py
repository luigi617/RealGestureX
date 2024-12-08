# scripts/inference.py

import cv2
import mediapipe as mp
import torch
from models.HandDetectionModel import get_finetuned_hand_detection_model, get_transformer
from models.StaticGestureModel import StaticGestureModel
from models.DynamicGestureModel import DynamicGestureModel
from utils.utils import preprocess_landmarks
from collections import deque
import numpy as np
from utils.utils import map_gesture_to_command
import time
from models.GestureClasses import static, dynamic
from PIL import Image



def extract_hand_bbox(result, frame):

    box = []

    if result.multi_hand_landmarks:
        for landmarks in result.multi_hand_landmarks:
            # Calculate bounding box of the hand
            x_min = min([landmark.x for landmark in landmarks.landmark])-0.03
            y_min = min([landmark.y for landmark in landmarks.landmark])-0.03
            x_max = max([landmark.x for landmark in landmarks.landmark])+0.03
            y_max = max([landmark.y for landmark in landmarks.landmark])+0.03

            # Convert normalized coordinates to pixel values
            h, w, _ = frame.shape
            x_min, y_min, x_max, y_max = int(x_min * w), int(y_min * h), int(x_max * w), int(y_max * h)

            box.append([x_min, y_min, x_max, y_max])

    return box

def recognize_gestures():
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load Static Gesture Model
    hand_detection_model = get_finetuned_hand_detection_model()
    hand_detection_model.to(device)
    hand_detection_model.eval()


    # Initialize MediaPipe Hands
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
    )
    mp_drawing = mp.solutions.drawing_utils


    

    # Start video capture
    cap = cv2.VideoCapture(2)
    prev_time = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # Flip the frame horizontally for a mirror effect
        frame = cv2.flip(frame, 1)
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(image)


        sorted_indices = torch.argsort(output[0]["scores"], descending=True)
        sorted_boxes = output[0]["boxes"][sorted_indices]
        for box in sorted_boxes[:3]:
            box = [int(x) for x in box.tolist()]
            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp.solutions.drawing_utils.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)


              

        # Display frame rate
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
        prev_time = curr_time
        cv2.putText(frame, f'FPS: {int(fps)}', (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

        # Display the resulting frame
        cv2.imshow('Gesture Recognition', frame)

        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    recognize_gestures()
