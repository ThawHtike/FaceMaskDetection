
import streamlit as st
from PIL import Image
import numpy as np
import cv2
import time
from detect_and_predict import detect_and_predict_mask

# --- CONFIGURATION ---
st.set_page_config(
    page_title='Real-Time Face Mask Detector',
    page_icon='😷',
    layout='centered',
    initial_sidebar_state='expanded'
)

# --- WEB APP TITLE & MENU ---
st.markdown('<h1 align="center">😷 Real-Time Face Mask Detection</h1>', unsafe_allow_html=True)

activities = ["Image Detection", "Webcam Detection"]
st.sidebar.markdown("# Select Mode")
choice = st.sidebar.selectbox("Choose the detection mode:", activities)

# ====================================================================
# MODE 1: IMAGE DETECTION
# ====================================================================
if choice == 'Image Detection':
    st.markdown('### Upload your image here ⬇️')

    image_file = st.file_uploader("", type=['jpg', 'jpeg', 'png'])

    if image_file is not None:
        # 1. Display uploaded image
        our_image = Image.open(image_file)
        st.image(our_image, caption='Uploaded Image', use_column_width=True)

        if st.button('Process Image'):
            st.markdown('### Detection Result 👇')

            # Convert PIL Image to OpenCV BGR format
            img_array = np.array(our_image.convert('RGB'))
            opencv_image_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

            # 2. Call the core detection function
            try:
                processed_bgr_image = detect_and_predict_mask(opencv_image_bgr)

                # Convert back to RGB for Streamlit display
                processed_rgb_image = cv2.cvtColor(processed_bgr_image, cv2.COLOR_BGR2RGB)

                # 3. Display the result
                st.image(processed_rgb_image, caption='Processed Image', use_column_width=True)
            except Exception as e:
                st.error(f"An error occurred during detection. Check if models are loaded. Error: {e}")

# ====================================================================
# MODE 2: WEBCAM DETECTION (Streamlit Integration)
# ====================================================================
elif choice == 'Webcam Detection':
    st.markdown('### Live Webcam Detection')
    st.info("Press 'Start Webcam' to begin real-time mask detection.")

    frame_placeholder = st.empty()
    cap = None  # Initialize cap outside the button check

    if st.button('Start Webcam'):
        cap = cv2.VideoCapture(0)  # 0 for default camera
        st.session_state['webcam_running'] = True

    if 'webcam_running' in st.session_state and st.session_state['webcam_running']:
        if st.button('Stop Webcam'):
            st.session_state['webcam_running'] = False

        while cap and st.session_state['webcam_running']:
            ret, frame = cap.read()
            if not ret:
                st.warning("Cannot read frame from webcam.")
                st.session_state['webcam_running'] = False
                break

            # Call the core detection function (BGR input, BGR output)
            processed_bgr_frame = detect_and_predict_mask(frame)

            # Convert to RGB for Streamlit display
            processed_rgb_frame = cv2.cvtColor(processed_bgr_frame, cv2.COLOR_BGR2RGB)

            # Update the placeholder image
            frame_placeholder.image(processed_rgb_frame, channels="RGB", use_column_width=True)

            # Control frame rate
            time.sleep(0.01)

        if cap:
            cap.release()
            frame_placeholder.empty()
            st.success("Webcam stream stopped.")

    # If the stop button was pressed externally or stream ended
    if cap and not ('webcam_running' in st.session_state and st.session_state['webcam_running']):
        cap.release()

# --- HOW TO RUN ---
# To run this Streamlit application, execute:
# streamlit run app.py