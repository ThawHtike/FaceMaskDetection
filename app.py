import streamlit as st
from PIL import Image
import numpy as np
import cv2
import time
from detect_and_predict import detect_and_predict_mask

# -------------------------------------------------------
# STREAMLIT CONFIG
# -------------------------------------------------------
st.set_page_config(
    page_title='Real-Time Face Mask Detector',
    page_icon='😷',
    layout='centered',
    initial_sidebar_state='expanded'
)

st.markdown('<h1 align="center">😷 Real-Time Face Mask Detection</h1>', unsafe_allow_html=True)

activities = ["Image Detection", "Webcam Detection"]
st.sidebar.markdown("# Select Mode")
choice = st.sidebar.selectbox("Choose the detection mode:", activities)

# =======================================================
# MODE 1: IMAGE DETECTION
# =======================================================
if choice == 'Image Detection':
    st.markdown('### Upload your image here ⬇️')

    image_file = st.file_uploader(
        "Upload Image",
        type=['jpg', 'jpeg', 'png'],
        label_visibility="collapsed"
    )

    if image_file is not None:
        # Display uploaded image
        our_image = Image.open(image_file)
        st.image(our_image, caption='Uploaded Image', use_column_width=True)

        if st.button('Process Image'):
            st.markdown('### Detection Result 👇')

            # Convert image to OpenCV format
            img_array = np.array(our_image.convert('RGB'))
            opencv_image_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

            try:
                # Run detection model
                processed_bgr_image = detect_and_predict_mask(opencv_image_bgr)
                processed_rgb_image = cv2.cvtColor(processed_bgr_image, cv2.COLOR_BGR2RGB)

                # Show processed output
                st.image(processed_rgb_image, caption='Processed Image', use_column_width=True)

            except Exception as e:
                st.error(f"Error during detection: {e}")


# =======================================================
# MODE 2: WEBCAM DETECTION
# =======================================================
elif choice == 'Webcam Detection':
    st.markdown('### Live Webcam Detection')
    st.info("Press 'Start Webcam' to begin real-time mask detection.")

    frame_placeholder = st.empty()

    # Initialize webcam state
    if "webcam_running" not in st.session_state:
        st.session_state.webcam_running = False

    # Start webcam
    if st.button("Start Webcam"):
        st.session_state.webcam_running = True
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            st.error("Could not access the webcam.")
            st.session_state.webcam_running = False
        else:
            st.success("Webcam started!")

            while st.session_state.webcam_running:
                ret, frame = cap.read()
                if not ret:
                    st.warning("Failed to capture frame.")
                    break

                # Run detection
                processed_bgr_frame = detect_and_predict_mask(frame)
                processed_rgb_frame = cv2.cvtColor(processed_bgr_frame, cv2.COLOR_BGR2RGB)

                # Display frame
                frame_placeholder.image(processed_rgb_frame, channels="RGB")

                time.sleep(0.02)

            cap.release()
            frame_placeholder.empty()
            st.success("Webcam stopped.")

    # Stop webcam
    if st.button("Stop Webcam"):
        st.session_state.webcam_running = False
