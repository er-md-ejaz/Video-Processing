# Real-Time Object Detection with YOLO, Flask Backend & Dashboard  

This project integrates **YOLOv8 object detection** with a **Flask backend** and a **dashboard** for visualization.  
It allows detecting objects in real-time from webcam/video, storing results in a database, and monitoring detections via a dashboard.  

---

## 🚀 Features
- Real-time object detection using **YOLOv8**  
- Flask-based backend with **SQLite database**  
- REST APIs for storing & fetching detection results  
- Interactive dashboard to view:  
  - Detected objects  
  - Confidence scores  
  - Timestamped detection history  
- Option to save detection video outputs  

---

## 📂 Project Structure
project/
│── server.py # Backend + Database + API + Dashboard
│── detector.py # YOLO client code for real-time detection
│── detections.db # SQLite database (auto-created on first run)
│── requirements.txt # Required dependencies
