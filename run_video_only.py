
from imutils.video import VideoStream
from detect_and_predict import detect_and_predict_mask # Import the core logic
import imutils
import time
import cv2

# initialize the video stream and allow the camera sensor to warm up
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)

# loop over the frames from the video stream
while True:
    # grab the frame from the threaded video stream and resize it
    frame = vs.read()
    # Resize frame for faster processing (standard practice)
    frame = imutils.resize(frame, width=800)

    # Detect faces and predict mask status
    processed_frame = detect_and_predict_mask(frame)

    # show the output frame
    cv2.imshow("Real-Time Face Mask Detection (OpenCV)", processed_frame)
    key = cv2.waitKey(1) & 0xFF

    # if the 'q' key was pressed, break from the loop
    if key == ord("q"):
        break

# cleanup
cv2.destroyAllWindows()
vs.stop()

# --- HOW TO RUN ---
# To run this script, execute:
# python run_video_only.py