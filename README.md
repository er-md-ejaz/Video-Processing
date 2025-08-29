# 🔍 Real-Time Object Detection with YOLO 🚀  
This project integrates **YOLOv8 object detection** with a **Flask backend** and an **interactive dashboard** 📊.  
It enables real-time object detection from webcam/video, saves detection results in a database, and displays them on a dashboard.  

---

## ✨ Features
✅ Real-time object detection using **YOLOv8**  
✅ Flask backend with **SQLite database**  
✅ REST API for storing & fetching detection results  
✅ Interactive **Dashboard** to view:  
   - 📌 Detected objects  
   - 📊 Confidence scores  
   - ⏱️ Timestamped detection history  
✅ Option to **save detection video outputs**  

---

## 📂 Project Structure

```plaintext
project/
│── server.py          # ⚙️ Backend + Database + API + Dashboard
│── detector.py        # 🎥 YOLO client code for real-time detection
│── detections.db      # 🗄️ SQLite database (auto-created on first run)
│── requirements.txt   # 📦 Required dependencies
