# Smart Door Lock System (Face Recognition)

This project is a Python-based smart door system that uses face recognition to verify users and simulate door unlocking.

## Features
- Face recognition authentication
- Webcam-based face capture
- Real-time face detection using OpenCV
- Face encodings saved in a .pkl file
- Very simple to run

## Technologies Used
Python
OpenCV
face_recognition
NumPy
Pickle

## Project Files
encode_faces.py  
face_capture.py  
face_encodings.pkl  
recognize_faces.py  
README.md  

(Note: dataset/ and faces/ folders are not uploaded because they contain personal images. Create them locally.)

## How It Works

1. Capture Face Images  
Run this in your terminal:  
python face_capture.py  
This will open the webcam and capture face images.

2. Generate Face Encodings  
Run:  
python encode_faces.py  
This will generate face_encodings.pkl.

3. Run Real-Time Recognition  
Run:  
python recognize_faces.py  
If your face matches the stored encoding → Access Granted  
Otherwise → Access Denied

## Installation
Install required packages:  
pip install opencv-python  
pip install face_recognition  
pip install numpy  
(Optional: pip install dlib)

## Author
Srushti Swami
