# Facial Recognition System

This project is a basic facial recognition system built using Python, OpenCV, and the `face_recognition` library. The system captures a live video feed, detects faces, and attempts to recognize them based on a set of preloaded known images.

## Features

- Detects and recognizes faces in a live video feed.
- Draws bounding boxes and labels for recognized faces.
- Uses pre-trained encodings for face recognition.

## Requirements

To run this project, you’ll need:

- **Python 3.6+**
- **OpenCV** for image and video processing
- **face_recognition** library for face detection and encoding
- A **webcam** or external camera for capturing the video feed

## Installation

Clone the repository and install the required libraries:

```bash
git clone https://github.com/yourusername/facial-recognition-system.git
cd facial-recognition-system
pip install opencv-python face-recognition
