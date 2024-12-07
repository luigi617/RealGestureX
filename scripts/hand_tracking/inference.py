# inference.py

import cv2
import os
from ultralytics import YOLO
import torch

def main():

    model_path = os.getcwd() + '/models/parameters/hand_tracking_model.pt'

    # Verify that the model file exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"YOLO model file not found at {model_path}")


    model = YOLO(model_path)

    cap = cv2.VideoCapture(2)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame. Exiting...")
            break

        # Optional: Flip the frame horizontally for a mirror effect
        frame = cv2.flip(frame, 1)

        # Perform inference using the YOLO model
        # The 'predict' method can be used, but for real-time inference, it's more efficient to use 'model' directly
        results = model(frame, verbose=False)[0]

        top_k = 3  # Number of top boxes to display

        # Check if any detections are made
        if results.boxes:
            # Extract confidence scores as a tensor
            confs = results.boxes.conf  # Tensor of shape [num_boxes]

            # Determine the number of boxes to display
            num_boxes = len(confs)
            k = min(top_k, num_boxes)

            # Get the top k indices based on confidence scores
            # torch.topk returns the top k values and their indices
            topk_conf, topk_indices = torch.topk(confs, k)

            # Select the top k boxes
            top_boxes = results.boxes[topk_indices]

            # Iterate through the top k boxes and draw them
            for box in top_boxes:
                # Extract bounding box coordinates
                x_min, y_min, x_max, y_max = box.xyxy.cpu().numpy()[0].astype(int)

                # Extract confidence score and class
                confidence = box.conf.cpu().numpy()[0]
                class_id = int(box.cls.cpu().numpy()[0])

                # Map class_id to class name (assuming 'hand' is class 0)
                class_name = "hand" if class_id == 0 else f"class_{class_id}"

                # Define the color for the bounding box and text
                color = (0, 255, 0)  # Green color in BGR

                # Draw the bounding box
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)

                # Prepare the label with class name and confidence
                label = f"{class_name} {confidence:.2f}"

                # Calculate label size
                (label_width, label_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

                # Draw a filled rectangle behind the label for better visibility
                cv2.rectangle(frame, (x_min, y_min - label_height - baseline), 
                              (x_min + label_width, y_min), color, thickness=cv2.FILLED)

                # Put the label text above the bounding box
                cv2.putText(frame, label, (x_min, y_min - baseline), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        else:
            # No detections, you can handle it if needed
            pass

        cv2.imshow("Hand Detection Inference", frame)

        # Define the key press actions
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("Quitting...")
            break
        # You can add more key actions if needed

    # ----------------------------
    # Step 5: Release Resources
    # ----------------------------

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
