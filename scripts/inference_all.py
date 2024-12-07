

import cv2
import os
from ultralytics import YOLO
import torch

def main():

    model_path = os.getcwd() + '/models/parameters/hand_tracking_model.pt'

    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"YOLO model file not found at {model_path}")


    model = YOLO(model_path)

    cap = cv2.VideoCapture(2)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame. Exiting...")
            break

        
        frame = cv2.flip(frame, 1)

        
        
        results = model(frame, verbose=False)[0]

        top_k = 3 
        if results.boxes:
            confs = results.boxes.conf
            num_boxes = len(confs)
            k = min(top_k, num_boxes)
            topk_conf, topk_indices = torch.topk(confs, k)
            top_boxes = results.boxes[topk_indices]
            for box in top_boxes:
                x_min, y_min, x_max, y_max = box.xyxy.cpu().numpy()[0].astype(int)
                confidence = box.conf.cpu().numpy()[0]
                class_id = int(box.cls.cpu().numpy()[0])
                class_name = "hand" if class_id == 0 else f"class_{class_id}"
                color = (0, 255, 0)
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)
                label = f"{class_name} {confidence:.2f}"
                (label_width, label_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(frame, (x_min, y_min - label_height - baseline), 
                              (x_min + label_width, y_min), color, thickness=cv2.FILLED)
                cv2.putText(frame, label, (x_min, y_min - baseline), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        else:
            
            pass

        cv2.imshow("Hand Detection Inference", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("Quitting...")
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
