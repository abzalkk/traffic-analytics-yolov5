# 🚦 Traffic Analytics System using YOLOv5

A real-time **highway traffic analytics system** that captures live FL511 camera feeds, detects vehicles using **YOLOv5**, and provides lane-level congestion insights to help determine which lane offers the fastest route during heavy traffic.

---

## 🧠 Overview
This project combines **computer vision**, **object detection**, and **real-time analytics** to analyze live traffic camera streams. Using **YOLOv5** for vehicle detection and **OpenCV** for frame processing, the system identifies and counts vehicles in each lane, providing insights on traffic flow and congestion patterns.

---

## ⚙️ Tech Stack
- **Python 3.11+**
- **YOLOv5 (Torch Hub)**
- **OpenCV**
- **MSS** for live screen capture
- **NumPy** for fast array processing
- **Quartz API (macOS)** for Chrome window targeting

---

## 🚀 Features
✅ Real-time screen capture from **FL511 Chrome tab**  
✅ YOLOv5-based **vehicle detection and classification** (cars, trucks, buses, motorcycles)  
✅ On-screen display of FPS, vehicle counts, and window info  
✅ **Lane segmentation overlay** to estimate congestion by lane  
✅ Configurable fullscreen view and adjustable region of interest  
✅ Compatible with multi-monitor setups  

---

## 📷 Example Output
*(Replace these with your screenshots)*

| Live Camera Feed | Detection Output |
|------------------|------------------|
| ![input](screenshots/fl511_input.png) | ![output](screenshots/fl511_output.png) |

---

## 🧩 Installation

1️⃣ Clone the repository:
```bash
git clone https://github.com/abzalkk/traffic-analytics-yolov5.git
cd traffic-analytics-yolov5
