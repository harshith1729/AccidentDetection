# 🚨 Guardian Connect – AI‑Powered Accident Detection Dashboard And SOS Using Twillo

An intelligent real‑time accident detection and emergency response system built using **YOLO object detection**, **computer vision**, and **Twilio‑based automated WhatsApp alerts**, wrapped in a rich **Streamlit dashboard**.

---

## 📋 Overview

**Guardian Connect** detects accidents in real‑time from video streams or uploaded files, analyzes severity using multiple visual cues (area, position, debris, fire), and automatically alerts emergency services (hospitals, police, and fire stations). The dashboard provides full visualization, analytics, and alert tracking.

---

## ✨ Key Features

### 🎯 Accident Detection

* YOLO‑based accident localization.
* Multi‑factor severity scoring (0‑12 scale): **Severe**, **Moderate**, **Minor**.
* Fire detection using HSV color analysis.
* Optional person detection for critical incidents.
* GPU acceleration with FP16 support.

### 📹 Dual Mode Operation

#### Camera Mode (Live Detection)

* Real‑time frame analysis with FPS monitoring.
* Sends WhatsApp alerts for each **Severe** detection (score ≥ 7).
* 15‑second cooldown to avoid alert spam.

#### Video Mode (Batch Analysis)

* Processes uploaded video files.
* Tracks maximum severity detection.
* Sends one summary alert at video completion.

### 🚨 Smart Alert System (Twilio Integration)

* Automatic WhatsApp alerts to emergency services.
* **Fire‑aware routing:**

  * No fire → Hospital + Police.
  * Fire detected → Hospital + Police + Fire Station.
* Location sent as a Google Maps link.
* Non‑blocking, asynchronous Twilio alert threads.
* Alert log with Pending, Sent, Failed states.

### 📊 Real‑Time Dashboard Analytics

* Live FPS, frame count, and detection count.
* Severity distribution metrics.
* Historical detection logs with timestamps.
* Interactive Folium map for location tracking.

---

## 🧠 How the Dashboard Works

### 🎛 Layout Overview

* **Header:** Displays title, subtitle, and project branding.
* **Sidebar:** Configure source (camera/upload), coordinates, and Twilio status.
* **Main Display:** Real‑time video stream with bounding boxes and severity labels.
* **Metrics Panel:** Tracks Frames, Detections, FPS, and system status.
* **Alert Panel:** Displays WhatsApp alert history.
* **Map Panel:** Shows accident location via Folium map marker.
* **Detection Summary:** Aggregates severity counts with detailed logs.

### 🧩 Core Components

| Component                     | Purpose                                                  |
| ----------------------------- | -------------------------------------------------------- |
| `UltraFastDetectionProcessor` | Loads YOLO model, processes frames at high FPS.          |
| `OptimizedSeverityClassifier` | Extracts area, contrast, and shape features for scoring. |
| `FastFireDetector`            | Detects fire regions using HSV color range.              |
| `TwilioAlertSystem`           | Handles WhatsApp alert queue, sending, and retries.      |

---

## 🛠️ Tech Stack

### Frontend / Detection System

* **Python 3.9+**
* **Streamlit** – Interactive web UI.
* **OpenCV** – Frame capture & preprocessing.
* **Ultralytics YOLO** – Object detection.
* **PyTorch** – Model execution.
* **Folium + Streamlit‑Folium** – Map visualization.
* **Twilio SDK** – WhatsApp messaging.

### Backend (Optional)

* **Node.js + Express** – Lightweight REST API server.
* **Supabase (PostgreSQL)** – Storage for amenities & alerts.

### Database Schema

**Tables:**

* `accident_alerts(id, latitude, longitude, description, created_at)`
* `amenities(id, name, type, lat, lon, email, address)`
* `alert_notifications(id, alert_id, amenity_id, distance_km, created_at)`

---

## 🚀 Installation & Setup

### 1️⃣ Prerequisites

* Python 3.9+
* Node.js 16+ (optional backend)
* Twilio WhatsApp sandbox or verified business number.

### 2️⃣ Clone Repository

```bash
git clone <repo_url>
cd Dashboard
```

### 3️⃣ Python Setup

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 4️⃣ Configure `.env`

```bash
TWILIO_ACCOUNT_SID=ACXXXXXXXXXXXXXXXXXXXXXXXXXXXX
TWILIO_AUTH_TOKEN=your_auth_token
TWILIO_WHATSAPP_FROM=whatsapp:+XXXXXXXXXX
EMERGENCY_NUMBER=whatsapp:+91XXXXXXXXXX
YOLO_MODEL_PATH=/path/to/models/best.pt
```

### 5️⃣ Optional Node Backend

```bash
npm install
node server.js
```

### 6️⃣ Run the Dashboard

```bash
streamlit run app.py
```

Then visit: [http://localhost:8501](http://localhost:8501)

---

## 📖 Usage

### Starting the System

* **Option 1:** Live Camera → Select index (0,1,...)
* **Option 2:** Upload video (mp4/avi/mov)
* Set GPS coordinates in sidebar.
* Click **🚀 Start Detection** to begin.

### Monitoring

* View real‑time detections and FPS.
* Severity color codes:

  * 🔴 Severe (8–12)
  * 🟠 Moderate (5–7)
  * 🟢 Minor (0–4)
* Alerts automatically sent for severe detections.

### Emergency Alerts

* Alerts include accident severity, confidence, and map link.
* Fire incidents trigger additional alert routing.
* Manual test alerts can be sent from the dashboard.

---

## 🧮 Severity Scoring Breakdown

| Factor            | Max Points | Description                    |
| ----------------- | ---------- | ------------------------------ |
| Area Ratio        | 4          | Large debris → higher severity |
| Road Position     | 3          | Center of road → dangerous     |
| Dark Debris       | 2          | Indicates fluids/blood         |
| Edge Density      | 2          | Scattered debris → impact      |
| Orientation       | 3          | Fallen vehicle/person          |
| Contrast Variance | 2          | Indicates damage               |
| Person Detection  | 8          | Automatic severe detection     |

---

## 🐛 Troubleshooting

* **Model Load Error:** Verify path & YOLO version.
* **Twilio Error:** Check sandbox linking & number format.
* **Low FPS:** Reduce frame size or enable GPU FP16.
* **No Alerts:** Ensure `.env` has correct Twilio creds.
* **Map Not Displaying:** Confirm internet connection.

---

## ⚡ Performance Notes

* CPU inference: ~15–25 FPS.
* GPU (FP16): ~60–120 FPS.
* Async Twilio worker avoids UI freezes.

---

## 📈 Future Enhancements

* Email + SMS notifications.
* Multi‑camera input.
* Historical alert analytics.
* Mobile app integration.
* Environmental data fusion (weather, traffic).

---

## 👥 Contributors

**Harshith** – Developer, Vision System & Twilio Integration
Acknowledgments: Ultralytics YOLO, OpenCV, Streamlit, Twilio API, Supabase.

---

OutPuts : 

Twillo Alerts :

<img width="1280" height="689" alt="Screenshot 2025-11-09 at 22 44 20" src="https://github.com/user-attachments/assets/29c6b415-5078-440b-a70d-3e27badc420b" />

<img width="1280" height="685" alt="Screenshot 2025-11-09 at 22 47 08" src="https://github.com/user-attachments/assets/ecbb8242-fd96-4ea8-a102-23e94a635d35" />

<img width="551" height="608" alt="Screenshot 2025-11-09 at 22 48 11" src="https://github.com/user-attachments/assets/a5fe4308-7d52-4117-bf7e-310153a76c6b" />


Dashboard Outputs : 

<img width="1280" height="685" alt="Screenshot 2025-11-09 at 22 49 40" src="https://github.com/user-attachments/assets/fd5d396b-77a7-41a5-840a-312ad0826fd0" />

<img width="1280" height="687" alt="Screenshot 2025-11-09 at 22 54 31" src="https://github.com/user-attachments/assets/c855b355-3e18-4721-9c38-01067a411a18" />










