# scripts/inference.py

import cv2
import mediapipe as mp
import torch
from models.StaticGestureModel import StaticGestureModel
from models.DynamicGestureModel import DynamicGestureModel
from utils.utils import preprocess_landmarks
from collections import deque
import numpy as np
from utils.gesture_commands import map_gesture_to_command
import time


static = [
    "pointing",
    "open_palm",
    "thumb_index_touch",
    "fist",
    "thumb_up",
    "thumb_down",
    "peace_sign",
    "crossed_finger",
    "shaka",
    "rock_on",
    "pinched_fingers",
]

dynamic = [
    "swipe_up",
    "swipe_down",
    "swipe_left",
    "swipe_right",
    "wave",
]



def recognize_gestures():
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load Static Gesture Model
    static_model = StaticGestureModel(input_size=63, num_classes=len(static))
    static_model.load_state_dict(torch.load('models/static_gesture_model.pth', map_location=device))
    static_model.to(device)
    static_model.eval()

    # Load Dynamic Gesture Model
    dynamic_model = DynamicGestureModel(num_classes=len(dynamic), hidden_size=128, num_layers=2)
    dynamic_model.load_state_dict(torch.load('models/dynamic_gesture_model.pth', map_location=device))
    dynamic_model.to(device)
    dynamic_model.eval()

    # Initialize MediaPipe Hands
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
    )
    mp_drawing = mp.solutions.drawing_utils


    # Initialize buffer for dynamic gestures
    sequence_length = 30  # Number of frames to consider for dynamic gestures
    buffer = deque(maxlen=sequence_length)

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

        gesture = None
        confidence = 0.0

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw landmarks on the frame
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Extract and preprocess landmarks
                landmarks = []
                for lm in hand_landmarks.landmark:
                    landmarks.extend([lm.x, lm.y, lm.z])
                processed_landmarks = preprocess_landmarks(landmarks).to(device)

                # Static Gesture Prediction
                with torch.no_grad():
                    static_output = static_model(processed_landmarks.unsqueeze(0))  # Shape: (1, num_classes)
                    static_probs = torch.softmax(static_output, dim=1)
                    static_confidence_val, static_pred = torch.max(static_probs, 1)
                    static_gesture = static[static_pred.item()]
                    static_confidence_val = static_confidence_val.item()

                # Add to buffer for dynamic gesture
                buffer.append(processed_landmarks.cpu().numpy())

                # Dynamic Gesture Prediction
                if len(buffer) == sequence_length:
                    dynamic_sequence = np.array(buffer)  # Shape: (seq_len, 63)
                    # Reshape to (1, seq_len, 63)
                    dynamic_sequence = torch.FloatTensor(dynamic_sequence).unsqueeze(0).to(device)

                    with torch.no_grad():
                        dynamic_output = dynamic_model(dynamic_sequence)  # Shape: (1, num_classes)
                        dynamic_probs = torch.softmax(dynamic_output, dim=1)
                        dynamic_confidence_val, dynamic_pred = torch.max(dynamic_probs, 1)
                        dynamic_gesture = dynamic[dynamic_pred.item()]
                        dynamic_confidence_val = dynamic_confidence_val.item()

                # Decide which gesture to prioritize
                if len(buffer) == sequence_length and dynamic_confidence_val > 0.8 and dynamic_confidence_val > static_confidence_val:
                    gesture = dynamic_gesture
                    confidence = dynamic_confidence_val
                else:
                    gesture = static_gesture
                    confidence = static_confidence_val

                # gesture = static_gesture
                # confidence = static_confidence_val

                # Map gesture to command
                if confidence > 0.6:  # Confidence threshold
                    command = map_gesture_to_command(gesture)
                    cv2.putText(frame, f'Gesture: {gesture} ({confidence*100:.1f}%)', (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                    # Execute command
                    # execute_command(command)
                else:
                    cv2.putText(frame, f'Gesture: None', (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

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
