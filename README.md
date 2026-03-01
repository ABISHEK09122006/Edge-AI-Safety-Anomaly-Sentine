# Edge-AI-Safety-Anomaly-Sentine
Edge AI based smart surveillance system for intrusion detection, human fall detection, and restricted zone monitoring using YOLOv8 and real-time video processing. Optimized for AMD-powered edge devices for low-latency smart city deployment.
# Edge AI Safety & Anomaly Sentinel

## Overview
Edge AI Safety & Anomaly Sentinel is a real-time smart surveillance system designed for smart city and smart building environments.  
The system performs:

- Intrusion Detection (Zone-based boundary crossing)
- Human Fall Detection
- Restricted Area Monitoring
- Real-time Alert Generation

Optimized for AMD-powered edge systems to ensure low-latency, privacy-preserving AI deployment.

---

## Problem Statement
Traditional surveillance systems only record footage but do not actively detect or prevent incidents in real-time.  
There is a need for intelligent, edge-deployable AI systems that:

- Detect intrusions instantly  
- Identify human falls (safety use case)  
- Monitor restricted zones  
- Work without cloud dependency  

---

## System Architecture

Camera → Frame Capture → YOLOv8 Detection →  
Zone Logic Engine → Event Classification →  
Real-Time Alert Overlay → Local Display / Alert Trigger  

---

## Technology Stack

- Python
- OpenCV
- YOLOv8 (Ultralytics)
- NumPy
- Real-time Edge Inference
- AMD Ryzen Multi-core CPU Optimization

---

## Key Features

- Real-time object detection
- Blue-line virtual boundary intrusion logic
- Human fall detection using bounding box aspect ratio analysis
- Multi-person detection handling
- Edge-based processing (no cloud required)
- Low-latency execution on AMD systems
- Scalable for smart city deployment

---

## Installation

```bash
pip install -r requirements.txt
python main.py
