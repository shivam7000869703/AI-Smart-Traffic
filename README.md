# AI Smart Traffic Management System 🚦

An intelligent 4-side intersection traffic monitoring and management system using YOLO11, OpenCV, and Streamlit.

## Features
- **Real-Time 4-Side Traffic Analysis**: North, South, East, and West intersection vehicle counts.
- **Vehicle Categorization**: Detects and tracks Cars, Motorcycles, Buses, and Trucks with YOLO11.
- **Dynamic Green Light Timer Calculation**: Calculates optimal signal timings and green light sequences based on real-time congestion.
- **Interactive Streamlit Dashboard**: Live monitoring, Image Analysis, Video Analysis, History & Reports, and IoT Integration tabs.
- **SQLite Database History**: Automatically logs detection records, timestamps, vehicle breakdowns, and signal sequences.

---

## Deploying on Render

This repository is pre-configured for one-click deployment on [Render](https://render.com/).

### Method 1: Using Render Blueprint (Recommended)
1. Fork or push this repository to your GitHub account: `shivam7000869703/AI-Smart-Traffic`.
2. Go to [Render Dashboard](https://dashboard.render.com/) and click **New +** -> **Blueprint**.
3. Connect your GitHub repository. Render will automatically read [`render.yaml`](./render.yaml).
4. Click **Apply**. Render will build and deploy your app.

### Method 2: Manual Web Service Setup
1. On [Render](https://dashboard.render.com/), click **New +** -> **Web Service**.
2. Connect your GitHub repository: `https://github.com/shivam7000869703/AI-Smart-Traffic`.
3. Configure the following fields:
   - **Environment**: `Python 3`
   - **Build Command**:
     ```bash
     pip install --upgrade pip && pip install -r requirements.txt
     ```
   - **Start Command**:
     ```bash
     streamlit run app.py --server.port $PORT --server.address 0.0.0.0 --server.headless true
     ```
   - **Plan**: `Free`
4. Under **Advanced** -> **Add Environment Variable**:
   - `PYTHON_VERSION`: `3.10.12`
   - `STREAMLIT_SERVER_HEADLESS`: `true`
   - `STREAMLIT_SERVER_ENABLE_CORS`: `false`
5. Click **Create Web Service**.

---

## Local Development

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

4. Run the Streamlit app:
```bash
streamlit run app.py
```
