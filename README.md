 😷 Face Mask Detection System (Real-Time)

A real-time computer vision project developed using Deep Learning (MobileNetV2) and OpenCV to automatically detect whether a person is wearing a face mask in live video streams or static images.

-----

 ✨ Features

   Real-Time Performance: Achieves high Frames Per Second (FPS) for smooth, live monitoring, specifically optimized to run efficiently on a standard CPU.
   High Accuracy: Uses the lightweight MobileNetV2 architecture for fast and accurate classification.
   Dual Output: Results are displayed in a clean, user-friendly Streamlit web interface or via a standalone OpenCV desktop window.
   Visual Feedback: Provides clear bounding boxes (Green for Mask, Red for No Mask) and prediction confidence scores.

 🛠️ Technologies Used

| Category | Tool / Library     | Purpose |
| :--- |:-------------------| :--- |
| Language | Python 3.9         | Core programming language. |
| Deep Learning | TensorFlow / Keras | Used to build, train, and load the mask classification model. |
| Model | MobileNetV2        | Lightweight CNN chosen for speed and efficiency on CPU. |
| Computer Vision | OpenCV             | Used for face detection (SSD) and video stream processing. |
| Interface | Streamlit          | Used to create the easy-to-use web application interface. |

-----

 ⚙️ Installation & Setup

To run this project locally, follow these steps:

 1. Clone the Repository

```bash
git clone [YOUR_GITHUB_REPO_URL]
cd face-mask-detection-project
```

 2. Create Virtual Environment (Recommended)

```bash
 For Windows/Linux
python -m venv .venv
source .venv/bin/activate
 For Windows PowerShell
 .venv\Scripts\Activate.ps1 
```

 3. Install Dependencies

Install all necessary libraries using the provided `requirements.txt` file.

```bash
pip install -r requirements.txt
```

 4. Get the Model Weights

   Ensure you have the trained model file, typically named `mask_detector.model`, located in the root project directory. (This file is often large and should be downloaded separately if not included in the repository).

-----

 🚀 Usage

You have two options for running the application:

 Option 1: Web Interface (Streamlit)

This is the easiest way to test with images or live webcam feed.

```bash
streamlit run app.py
```

This command will open the application in your default web browser (usually at `http://localhost:8501`).

 Option 2: Standalone Video Script (OpenCV)

Use this script for direct, fast, CPU-optimized testing and to measure the actual FPS.

```bash
python run_video_only.py
```

This will open a dedicated desktop window showing the live webcam feed with results.

 📊 Results & Performance

| Metric | Result | Discussion |
| :--- | :--- | :--- |
| Accuracy | 90-95% | The model achieves high accuracy on masked and unmasked faces (Chowdary et al., 2020). |
| Processing Speed | 10–15 FPS | Achieved real-time performance on a standard CPU by using the lightweight MobileNetV2 architecture (Sandler et al., 2018). |
| Model Size | Small/Lightweight | Optimized for quick loading and deployment. |

-----

 🤝 Future Work

   Improve Face Detector: Implement a single-stage model (like a customized YOLO) to eliminate Error Compounding issues (Nagrath et al., 2021).
   Alert System: Add an automatic alert feature (e.g., email or SMS) when a "No Mask" event is detected.
   Docker Deployment: Package the entire system using Docker for easier deployment in production environments.

 👥 Group Members

| Student Name | Student ID | Email |
| :--- | :--- | :--- |
| Muhammad Hashim (Group Leader) | 2430140035 | muhammadhashim@ogrenci.beykoz.edu.tr |
| Noman Ur Rehman | 2530150059 | nomanurrehman@ogrenci.beykoz.edu.tr |
| Thaw Zin Htike | 2430210021 | thawzinhtike@ogrenci.beykoz.edu.tr |
| Mohd Mudabbir | 2530210015 | mohdmudabbir@ogrenci.beykoz.edu.tr |

-----