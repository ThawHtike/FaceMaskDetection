
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import os

# --- Model Loading (Load once for efficiency) ---
try:
    # Load the pre-trained face detector model (Caffe model)
    print("[INFO] Loading face detector model...")
    prototxtPath = os.path.sep.join(["face_detector", "deploy.prototxt"])
    weightsPath = os.path.sep.join(["face_detector", "res10_300x300_ssd_iter_140000.caffemodel"])
    FACE_NET = cv2.dnn.readNet(prototxtPath, weightsPath)

    # Load the trained mask detector model (Keras model)
    print("[INFO] Loading face mask detector model...")
    MASK_NET = load_model("mask_detector.model")
except Exception as e:
    print(f"[ERROR] Could not load detection models: {e}")
    FACE_NET = None
    MASK_NET = None


def detect_and_predict_mask(frame):
    """
    Performs face detection and mask prediction on a single video frame or image.

    Args:
        frame: The input image/frame (OpenCV BGR format).

    Returns:
        The processed frame with bounding boxes (OpenCV BGR format).
    """
    if FACE_NET is None or MASK_NET is None:
        return frame  # Return original frame if models didn't load

    (h, w) = frame.shape[:2]
    # Construct a blob from the frame for Caffe model
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))

    FACE_NET.setInput(blob)
    detections = FACE_NET.forward()

    # Iterate over all detected faces
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > 0.5:  # Filter weak detections
            # Compute the (x, y)-coordinates of the bounding box
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # Ensure boxes are within frame dimensions
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

            face = frame[startY:endY, startX:endX]

            # Skip invalid/tiny faces
            if face.shape[0] < 20 or face.shape[1] < 20:
                continue

            # Preprocess the face for the mask prediction model
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)  # Convert to RGB
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)
            face = np.expand_dims(face, axis=0)

            # Make prediction
            (mask, withoutMask) = MASK_NET.predict(face, verbose=0)[0]

            # Determine label and color
            label = "Mask" if mask > withoutMask else "No Mask"
            color = (0, 255, 0) if label == "Mask" else (0, 0, 255)  # Green or Red (BGR)

            # Include probability in the label
            label = f"{label}: {max(mask, withoutMask) * 100:.2f}%"

            # Draw the label and bounding box on the original frame
            cv2.putText(frame, label, (startX, startY - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

    return frame