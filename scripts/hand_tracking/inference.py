

import cv2
import os
from ultralytics import YOLO
import torch
import logging
import warnings

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


    
    if torch.backends.mps.is_available():
        device = 'mps'
        logger.info("Using MPS (Apple GPU) device for inference.")
    else:
        device = 'cpu'
        logger.warning("MPS device not available. Using CPU for inference.")

    model_path = os.path.join(os.getcwd(), 'models', 'parameters', 'best.pt')
    if not os.path.exists(model_path):
        logger.error(f"YOLO model file not found at {model_path}")
        raise FileNotFoundError(f"YOLO model file not found at {model_path}")

    model = YOLO(model_path)
    model.to(device)
    cap = cv2.VideoCapture(2)


    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.warning("Failed to grab frame. Exiting...")
                break
            frame = cv2.flip(frame, 1)

            original_height, original_width = frame.shape[:2]
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            with torch.no_grad():
                results = model(frame_rgb, verbose=False)[0]
            print(len(results.boxes))
            top_k = 5
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
