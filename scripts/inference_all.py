

import cv2
import os
from ultralytics import YOLO
import torch
import logging
import warnings
from PIL import Image

from models.DynamicGestureCNNModel import DynamicGestureCNNModel
from models.StaticGestureCNNModel import StaticGestureCNNModel
from utils.utils import get_device

from models.GestureClasses import static, dynamic
from torchvision import transforms
from collections import deque


def setup_logging():
    """Configure logging for the script."""
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] [%(levelname)s] %(message)s',
        handlers=[
            logging.StreamHandler()
        ]
    )




def main():
    setup_logging()
    logger = logging.getLogger(__name__)
    
    device = get_device()

    model_path = os.path.join(os.getcwd(), 'models', 'parameters', 'hand_tracking_model.pt')
    if not os.path.exists(model_path):
        logger.error(f"YOLO model file not found at {model_path}")
        raise FileNotFoundError(f"YOLO model file not found at {model_path}")

    model = YOLO(model_path)
    model.to(device)



    static_model = StaticGestureCNNModel(num_classes=len(static))
    static_model.load_state_dict(torch.load('models/parameters/static_gesture_cnn_model.pth', map_location=device))
    static_model.to(device)
    static_model.eval()

    # Load Dynamic Gesture Model
    dynamic_model = DynamicGestureCNNModel(num_classes=len(dynamic), hidden_size=256, num_layers=2, bidirectional=True, freeze_cnn=True)
    dynamic_model.load_state_dict(torch.load('models/parameters/dynamic_gesture_cnn_model.pth', map_location=device))
    dynamic_model.to(device)
    dynamic_model.eval()

    transformer = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)), 
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    cap = cv2.VideoCapture(1)

    sequence_length = 15  # Number of frames to consider for dynamic gestures
    buffer = deque(maxlen=sequence_length)
    last_dynamic_gesture = (None, 0.0)
    last_static_gesture = (None, 0.0)
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.warning("Failed to grab frame. Exiting...")
                break
            frame = cv2.flip(frame, 1)

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)

            with torch.no_grad():
                results = model(pil_image, verbose=False)[0]
            top_k = 1
            if hasattr(results, 'boxes') and results.boxes:
                
                confs = results.boxes.conf  

                num_boxes = len(confs)
                k = min(top_k, num_boxes)
                topk_conf, topk_indices = torch.topk(confs, k)
                top_boxes = results.boxes[topk_indices]

                for box in top_boxes:
                    bbox = box.xyxy.cpu().numpy()
                    if bbox.ndim == 2 and bbox.shape[0] == 1:
                        x_min, y_min, x_max, y_max = bbox[0].astype(int)
                    else:
                        logger.warning("Unexpected bbox shape. Skipping this box.")
                        continue

                    
                    confidence = box.conf.cpu().numpy()
                    if confidence.size == 1:
                        confidence = confidence.item()
                    else:
                        logger.warning("Confidence is not a single value. Skipping this box.")
                        continue

                    class_id = box.cls.cpu().numpy()
                    if class_id.size == 1:
                        class_id = int(class_id.item())
                    else:
                        logger.warning("Class ID is not a single value. Skipping this box.")
                        continue

                    
                    class_name = "hand" if class_id == 0 else f"class_{class_id}"

                    
                    color = (0, 255, 0)  

                    
                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)
                    
                    label = f"{class_name} {confidence:.2f}"

                    (label_width, label_height), baseline = cv2.getTextSize(
                        label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
                    )
                    
                    cv2.rectangle(
                        frame,
                        (x_min, y_min - label_height - baseline),
                        (x_min + label_width, y_min),
                        color,
                        thickness=cv2.FILLED
                    )

                    cv2.putText(
                        frame, label, (x_min, y_min - baseline),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1
                    )




                    cropped_frame = frame_rgb[y_min:y_max, x_min:x_max].copy()
                    transformed_cropped_frame = transformer(cropped_frame)

                    dynamic_gesture = None
                    dynamic_confidence_val = 0.0
                    static_gesture = None
                    static_confidence_val = 0.0
                    with torch.no_grad():
                        static_output = static_model(transformed_cropped_frame.unsqueeze(0).to(device))
                        # static_confidence_val, static_pred = torch.max(static_output.data, 1)
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
                        last_static_gesture = (static_gesture, static_confidence_val)


                    gesture = static_gesture
                    confidence = static_confidence_val

                    cv2.putText(frame, f'Gesture: {gesture} ({confidence*100:.1f}%)', (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                    cv2.putText(frame, f'Last Dyanmic Gesture: {last_dynamic_gesture[0]} ({last_dynamic_gesture[1]*100:.1f}%)',
                                (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                    cv2.putText(frame, f'Last Static Gesture: {last_static_gesture[0]} ({last_static_gesture[1]*100:.1f}%)',
                                (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            else:
                logger.debug("No detections in this frame.")
                pass

            cv2.imshow("Hand Detection Inference", frame)

            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                logger.info("Quitting...")
                break
            

    except KeyboardInterrupt:
        logger.info("Interrupted by user. Exiting...")
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
    finally:
        
        
        
        cap.release()
        cv2.destroyAllWindows()
        logger.info("Resources released. Program terminated.")

if __name__ == "__main__":
    main()
