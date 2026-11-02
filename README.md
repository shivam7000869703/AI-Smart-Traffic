# AI Smart Traffic Management System

An intelligent traffic monitoring and management system using YOLO object detection, OpenCV, and Streamlit.

## Features
- **Real-time Vehicle Detection & Tracking**: Powered by YOLO11.
- **Traffic Density & Flow Analysis**: Automated calculation of vehicle counts and traffic congestion levels.
- **Interactive Dashboard**: Built with Streamlit for live monitoring, alerts, and historical reporting.
- **Database Logging**: Persistent traffic records with SQLite.

## Installation

1. Clone the repository:
```bash
git clone https://github.com/shivam7000869703/AI-Smart-Traffic.git
cd AI-Smart-Traffic
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Running the Application

To run the Streamlit dashboard:
```bash
streamlit run app.py
```
or
```bash
streamlit run dashboard_v2.py
```
