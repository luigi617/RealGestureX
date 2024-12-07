# inference.py

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

def suppress_specific_warnings():
    """
    Suppress specific warnings related to MPS backend limitations.
    """
    warnings.filterwarnings(
        "ignore",
        message="MPS: nonzero op is supported natively starting from macOS 14.0.*"
    )

def select_camera(max_attempts=5):
    """
    Attempt to select an active camera.

    Parameters:
        max_attempts (int): Number of camera indices to try.

    Returns:
        cap (cv2.VideoCapture): The video capture object if successful.

    Raises:
        RuntimeError: If no camera is found within the attempts.
    """
    for index in [2]:
        cap = cv2.VideoCapture(index)
        if cap.isOpened():
            logging.info(f"Using camera at index {index}.")
            return cap
        else:
            logging.warning(f"Camera at index {index} not accessible.")
            cap.release()
    raise RuntimeError(f"No accessible camera found in indices 0 to {max_attempts-1}.")

def main():
    setup_logging()
    logger = logging.getLogger(__name__)

    suppress_specific_warnings()

    # Determine the device to use: 'mps' for Apple GPU, else 'cpu'
    if torch.backends.mps.is_available():
        device = 'mps'
        logger.info("Using MPS (Apple GPU) device for inference.")
    else:
        device = 'cpu'
        logger.warning("MPS device not available. Using CPU for inference.")

    # Define the model path
    model_path = os.path.join(os.getcwd(), 'models', 'parameters', 'hand_tracking_model.pt')

    # Verify that the model file exists
    if not os.path.exists(model_path):
        logger.error(f"YOLO model file not found at {model_path}")
        raise FileNotFoundError(f"YOLO model file not found at {model_path}")

    # Load the YOLO model on the specified device
    try:
        model = YOLO(model_path)
        model.to(device)
        logger.info(f"Model loaded and moved to {device} device.")
    except Exception as e:
        logger.error(f"Failed to load the YOLO model: {e}")
        raise e

    # Attempt to select an active camera
    try:
        cap = select_camera()
    except RuntimeError as e:
        logger.error(e)
        return

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.warning("Failed to grab frame. Exiting...")
                break

            # Optional: Flip the frame horizontally for a mirror effect
            frame = cv2.flip(frame, 1)

            # Capture original frame dimensions
            original_height, original_width = frame.shape[:2]

            # Convert frame to RGB as YOLO expects images in RGB format
            try:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            except Exception as e:
                logger.error(f"Failed to convert frame to RGB: {e}")
                continue

            # Perform inference using the YOLO model
            try:
                results = model(frame_rgb, verbose=False)[0]
            except Exception as e:
                logger.error(f"Model inference failed: {e}")
                continue

            top_k = 5  # Number of top boxes to display

            # Check if any detections are made
            if hasattr(results, 'boxes') and results.boxes:
                # Extract confidence scores as a tensor
                confs = results.boxes.conf  # Tensor of shape [num_boxes]

                if confs is None or confs.numel() == 0:
                    logger.debug("No confidence scores found in detections.")
                    pass  # No detections
                else:
                    # Determine the number of boxes to display
                    num_boxes = len(confs)
                    k = min(top_k, num_boxes)

                    # Get the top k indices based on confidence scores
                    try:
                        topk_conf, topk_indices = torch.topk(confs, k)
                    except Exception as e:
                        logger.error(f"Error during topk operation: {e}")
                        continue

                    # Select the top k boxes
                    top_boxes = results.boxes[topk_indices]

                    # Iterate through the top k boxes and draw them
                    for box in top_boxes:
                        try:
                            # Extract bounding box coordinates
                            bbox = box.xyxy.cpu().numpy()
                            if bbox.ndim == 2 and bbox.shape[0] == 1:
                                x_min, y_min, x_max, y_max = bbox[0].astype(int)
                            else:
                                logger.warning("Unexpected bbox shape. Skipping this box.")
                                continue

                            # Extract confidence score and class using .item() to get scalar values
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

                            # Map class_id to class name (assuming 'hand' is class 0)
                            class_name = "hand" if class_id == 0 else f"class_{class_id}"

                            # Define the color for the bounding box and text
                            color = (0, 255, 0)  # Green color in BGR

                            # Draw the bounding box
                            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)

                            # Prepare the label with class name and confidence
                            label = f"{class_name} {confidence:.2f}"

                            # Calculate label size
                            (label_width, label_height), baseline = cv2.getTextSize(
                                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
                            )

                            # Draw a filled rectangle behind the label for better visibility
                            cv2.rectangle(
                                frame,
                                (x_min, y_min - label_height - baseline),
                                (x_min + label_width, y_min),
                                color,
                                thickness=cv2.FILLED
                            )

                            # Put the label text above the bounding box
                            cv2.putText(
                                frame, label, (x_min, y_min - baseline),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1
                            )
                        except Exception as e:
                            logger.error(f"Error processing a bounding box: {e}")
                            continue
            else:
                # No detections, you can handle it if needed
                logger.debug("No detections in this frame.")
                pass

            cv2.imshow("Hand Detection Inference", frame)

            # Define the key press actions
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                logger.info("Quitting...")
                break
            # You can add more key actions if needed

    except KeyboardInterrupt:
        logger.info("Interrupted by user. Exiting...")
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
    finally:
        # ----------------------------
        # Step 5: Release Resources
        # ----------------------------
        cap.release()
        cv2.destroyAllWindows()
        logger.info("Resources released. Program terminated.")

if __name__ == "__main__":
    main()
