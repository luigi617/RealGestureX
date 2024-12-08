# scripts/inference.py

import cv2
import mediapipe as mp
import torch
from models.StaticGestureCNNModel import StaticGestureCNNModel
from models.DynamicGestureCNNModel import DynamicGestureCNNModel
from utils.utils import map_gesture_to_command
import time
from models.GestureClasses import static, dynamic
from torchvision import transforms
from collections import deque



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
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    static_model = StaticGestureCNNModel(num_classes=len(static))
    static_model.load_state_dict(torch.load('models/parameters/static_gesture_cnn_model.pth', map_location=device))
    static_model.to(device)
    static_model.eval()

    # Load Dynamic Gesture Model
    dynamic_model = DynamicGestureCNNModel(num_classes=len(dynamic), hidden_size=256, num_layers=2, bidirectional=True, freeze_cnn=True)
    dynamic_model.load_state_dict(torch.load('models/parameters/dynamic_gesture_cnn_model.pth', map_location=device))
    dynamic_model.to(device)
    dynamic_model.eval()

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((128, 128)), 
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
    )
    mp_drawing = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(2)
    prev_time = 0

    sequence_length = 15  # Number of frames to consider for dynamic gestures
    buffer = deque(maxlen=sequence_length)
    last_dynamic_gesture = (None, 0.0)
    last_static_gesture = (None, 0.0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
        prev_time = curr_time
        cv2.putText(frame, f'FPS: {int(fps)}', (10, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp.solutions.drawing_utils.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        hand_data = extract_hand_bbox(results, frame_rgb)
        if hand_data:
            x_min, y_min, x_max, y_max = hand_data[0]
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

            cropped_frame = frame[y_min:y_max, x_min:x_max].copy()
            transformed_cropped_frame = transform(cropped_frame).unsqueeze(0).to(device)

            dynamic_gesture = None
            dynamic_confidence_val = 0.0
            static_gesture = None
            static_confidence_val = 0.0
            with torch.no_grad():
                static_output = static_model(transformed_cropped_frame)
                static_probs = torch.softmax(static_output, dim=1)
                static_confidence_val, static_pred = torch.max(static_probs, 1)
                static_gesture = static[static_pred.item()]
                static_confidence_val = static_confidence_val.item()
            
            buffer.append(transformed_cropped_frame.squeeze(0))

            if len(buffer) == sequence_length:
                dynamic_sequence = torch.stack(list(buffer)).unsqueeze(0).to(device)

                with torch.no_grad():
                    dynamic_output = dynamic_model(dynamic_sequence)
                    dynamic_probs = torch.softmax(dynamic_output, dim=1)
                    dynamic_confidence_val, dynamic_pred = torch.max(dynamic_probs, 1)
                    dynamic_gesture = dynamic[dynamic_pred.item()]
                    dynamic_confidence_val = dynamic_confidence_val.item()
            
            gesture = None
            confidence = 0.0
            if dynamic_gesture and dynamic_gesture not in ["not_moving", "moving_slowly"]:
                gesture = dynamic_gesture
                confidence = dynamic_confidence_val
                buffer.clear()
                last_dynamic_gesture = (dynamic_gesture, dynamic_confidence_val)
            else:
                gesture = static_gesture
                confidence = static_confidence_val
                last_static_gesture = (dynamic_gesture, dynamic_confidence_val)

            gesture = static_gesture
            confidence = static_confidence_val

            cv2.putText(frame, f'Gesture: {gesture} ({confidence*100:.1f}%)', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(frame, f'Last Dyanmic Gesture: {last_dynamic_gesture[0]} ({last_dynamic_gesture[1]*100:.1f}%)',
                        (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(frame, f'Last Static Gesture: {last_static_gesture[0]} ({last_static_gesture[1]*100:.1f}%)',
                        (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            if confidence > 0.6:
                command = map_gesture_to_command(gesture)

        cv2.imshow('Gesture Recognition', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    recognize_gestures()
