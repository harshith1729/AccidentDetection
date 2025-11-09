import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import torch
from collections import Counter, deque
import requests
from datetime import datetime
from urllib.parse import quote
from dotenv import load_dotenv
import folium
from streamlit_folium import st_folium
import threading
import queue
from PIL import Image
import io
import base64
import time
import os
from twilio.rest import Client

# Load environment variables
load_dotenv()

# Page Configuration
st.set_page_config(
    page_title="🚨 High-Speed Accident Detection",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); }
    .stApp { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); }
    .metric-card {
        background: rgba(255, 255, 255, 0.95);
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.2);
        margin: 10px 0;
    }
    .alert-critical {
        background: linear-gradient(135deg, #ff0844 0%, #ffb199 100%);
        color: white;
        padding: 20px;
        border-radius: 15px;
        font-weight: bold;
        font-size: 1.2em;
        text-align: center;
        animation: pulse 2s infinite;
    }
    @keyframes pulse {
        0%, 100% { transform: scale(1); }
        50% { transform: scale(1.05); }
    }
    .alert-success {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        color: white;
        padding: 15px;
        border-radius: 10px;
        font-weight: bold;
        margin: 10px 0;
    }
    .alert-pending {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 15px;
        border-radius: 10px;
        font-weight: bold;
        margin: 10px 0;
    }
    .header-container {
        background: rgba(255, 255, 255, 0.95);
        padding: 30px;
        border-radius: 20px;
        margin-bottom: 30px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
    }
    .video-container {
        background: rgba(0, 0, 0, 0.8);
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    }
</style>
""", unsafe_allow_html=True)

# Configuration
YOLO_MODEL_PATH = "/Users/harshith/Downloads/Projects/accidentDetection/best.pt"
DEFAULT_LAT = 17.3850
DEFAULT_LON = 78.4867

# Twilio Configuration
TWILIO_ACCOUNT_SID = os.getenv('TWILIO_ACCOUNT_SID')
TWILIO_AUTH_TOKEN = os.getenv('TWILIO_AUTH_TOKEN')
TWILIO_WHATSAPP_FROM = os.getenv('TWILIO_WHATSAPP_FROM')
EMERGENCY_NUMBER = os.getenv('EMERGENCY_NUMBER')

# Session state initialization
if 'streaming' not in st.session_state:
    st.session_state.streaming = False
if 'cap' not in st.session_state:
    st.session_state.cap = None
if 'detections' not in st.session_state:
    st.session_state.detections = []
if 'frame_count' not in st.session_state:
    st.session_state.frame_count = 0
if 'detection_count' not in st.session_state:
    st.session_state.detection_count = 0
if 'hospitals' not in st.session_state:
    st.session_state.hospitals = []
if 'stop_streaming' not in st.session_state:
    st.session_state.stop_streaming = False
if 'detection_history' not in st.session_state:
    st.session_state.detection_history = deque(maxlen=10)
if 'alert_sent' not in st.session_state:
    st.session_state.alert_sent = False
if 'alert_response' not in st.session_state:
    st.session_state.alert_response = None
if 'current_lat' not in st.session_state:
    st.session_state.current_lat = DEFAULT_LAT
if 'current_lon' not in st.session_state:
    st.session_state.current_lon = DEFAULT_LON
if 'alert_queue' not in st.session_state:
    st.session_state.alert_queue = queue.Queue()
if 'alert_status' not in st.session_state:
    st.session_state.alert_status = []
# NEW: Add alert status placeholder for real-time updates
if 'alert_status_container' not in st.session_state:
    st.session_state.alert_status_container = None

# ==========================================
# TWILIO ALERT SYSTEM (WITH UI UPDATES)
# ==========================================
class TwilioAlertSystem:
    """Non-blocking Twilio WhatsApp alert system with UI updates"""
    
    def __init__(self):
        self.client = None
        self.alert_thread = None
        self.running = False
        self.alert_queue = queue.Queue()
        self.sent_alerts = set()
        self.cooldown_time = 30
        self.last_alert_time = 0
        
        try:
            if TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN:
                self.client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
                st.success("✅ Twilio connected successfully!")
            else:
                st.warning("⚠️ Twilio credentials not found in .env file")
        except Exception as e:
            st.error(f"❌ Twilio initialization error: {e}")
    
    def start_alert_worker(self):
        """Start background alert sending thread"""
        if not self.running:
            self.running = True
            self.alert_thread = threading.Thread(target=self._alert_worker, daemon=True)
            self.alert_thread.start()
    
    def stop_alert_worker(self):
        """Stop background alert thread"""
        self.running = False
        if self.alert_thread:
            self.alert_thread.join(timeout=1)
    
    def _alert_worker(self):
        """Background worker that sends alerts without blocking main thread"""
        while self.running:
            try:
                alert_data = self.alert_queue.get(timeout=0.5)
                
                if alert_data:
                    self._send_whatsapp_alert(**alert_data)
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Alert worker error: {e}")
    
    def _send_whatsapp_alert(self, lat, lon, severity, score, has_fire, reasons):
        """Send WhatsApp alert via Twilio with UI updates"""
        try:
            # Check cooldown
            current_time = time.time()
            if current_time - self.last_alert_time < self.cooldown_time:
                return
            
            if not self.client or not EMERGENCY_NUMBER:
                return
            
            # Create alert signature
            alert_sig = f"{lat:.4f}_{lon:.4f}_{severity}"
            if alert_sig in self.sent_alerts:
                return
            
            # Build message
            maps_link = f"https://www.google.com/maps?q={lat},{lon}"
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            message_body = f"""🚨 *ACCIDENT DETECTED* 🚨

📍 Location: {maps_link}

⚠️ *Severity*: {severity}
📊 *Score*: {score}/12
🔥 *Fire*: {'YES - IMMEDIATE RESPONSE NEEDED!' if has_fire else 'No'}

🕒 Time: {timestamp}

📋 *Analysis*:
{chr(10).join(f'• {reason}' for reason in reasons[:3])}

⚡ Emergency services have been notified.
"""
            
            # Send via Twilio
            message = self.client.messages.create(
                from_=TWILIO_WHATSAPP_FROM,
                body=message_body,
                to=EMERGENCY_NUMBER
            )
            
            # Mark as sent
            self.sent_alerts.add(alert_sig)
            self.last_alert_time = current_time
            
            # **FIX: Update session state to trigger UI update**
            status = {
                'timestamp': timestamp,
                'severity': severity,
                'status': 'sent',
                'sid': message.sid,
                'has_fire': has_fire
            }
            st.session_state.alert_status.append(status)
            st.session_state.alert_sent = True
            
            print(f"✅ WhatsApp alert sent! SID: {message.sid}")
            
        except Exception as e:
            print(f"❌ Failed to send WhatsApp alert: {e}")
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            status = {
                'timestamp': timestamp,
                'severity': severity,
                'status': 'failed',
                'error': str(e),
                'has_fire': has_fire
            }
            st.session_state.alert_status.append(status)
    
    def queue_alert(self, lat, lon, severity, score, has_fire, reasons):
        """Queue an alert for sending (non-blocking)"""
        alert_data = {
            'lat': lat,
            'lon': lon,
            'severity': severity,
            'score': score,
            'has_fire': has_fire,
            'reasons': reasons
        }
        self.alert_queue.put(alert_data)
        
        # **FIX: Immediately add pending status to UI**
        pending_status = {
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'severity': severity,
            'status': 'pending',
            'has_fire': has_fire
        }
        st.session_state.alert_status.append(pending_status)

# ==========================================
# OPTIMIZED SEVERITY CLASSIFIER
# ==========================================
class OptimizedSeverityClassifier:
    """Ultra-fast severity classifier"""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def extract_features_fast(self, crop_img, bbox_coords, img_shape):
        """OPTIMIZED: Extract features with minimal processing"""
        features = {}
        
        x1, y1, x2, y2 = bbox_coords
        bbox_area = (x2 - x1) * (y2 - y1)
        img_area = img_shape[0] * img_shape[1]
        features['area_ratio'] = bbox_area / img_area
        
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        img_center_x = img_shape[1] / 2
        features['distance_from_center'] = abs(center_x - img_center_x) / img_shape[1]
        features['vertical_position'] = center_y / img_shape[0]
        
        small_crop = cv2.resize(crop_img, (64, 64))
        
        hsv = cv2.cvtColor(small_crop, cv2.COLOR_BGR2HSV)
        dark_mask = cv2.inRange(hsv, np.array([0, 0, 0]), np.array([180, 255, 50]))
        features['dark_ratio'] = np.count_nonzero(dark_mask) / dark_mask.size
        
        gray = cv2.cvtColor(small_crop, cv2.COLOR_BGR2GRAY)
        features['brightness_variance'] = np.var(gray.astype(np.float32))
        
        edges = cv2.Canny(gray, 50, 150)
        features['edge_density'] = np.count_nonzero(edges) / edges.size
        
        width = x2 - x1
        height = y2 - y1
        features['aspect_ratio'] = width / height if height > 0 else 1
        
        return features
    
    def detect_person_on_road_fast(self, crop_img):
        """OPTIMIZED: Fast person detection"""
        small_crop = cv2.resize(crop_img, (64, 64))
        hsv = cv2.cvtColor(small_crop, cv2.COLOR_BGR2HSV)
        
        dark_mask = cv2.inRange(hsv, np.array([0, 0, 0]), np.array([180, 255, 80]))
        dark_ratio = np.count_nonzero(dark_mask) / dark_mask.size
        
        if dark_ratio > 0.15:
            contours, _ = cv2.findContours(dark_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(largest_contour)
                compactness = (w * h) / (small_crop.shape[0] * small_crop.shape[1])
                
                if compactness > 0.3:
                    return True, dark_ratio, compactness
        
        return False, dark_ratio, 0
    
    def rule_based_severity_fast(self, features, crop_img):
        """OPTIMIZED: Fast rule-based classification"""
        score = 0
        reasons = []
        
        is_person, dark_ratio, compactness = self.detect_person_on_road_fast(crop_img)
        if is_person:
            score += 8
            reasons.append("CRITICAL: Person detected on road")
        
        if features['area_ratio'] > 0.12:
            score += 3
            reasons.append("Large accident area")
        elif features['area_ratio'] > 0.06:
            score += 2
        else:
            score += 1
        
        if features['distance_from_center'] < 0.2:
            score += 3
            reasons.append("Accident in center of road")
        elif features['distance_from_center'] < 0.35:
            score += 1
        
        if features['dark_ratio'] > 0.2:
            score += 2
            reasons.append("Significant dark debris detected")
        elif features['dark_ratio'] > 0.1:
            score += 1
        
        if features['edge_density'] > 0.12:
            score += 2
            reasons.append("Scattered debris detected")
        
        if 1.3 < features['aspect_ratio'] < 3.0:
            score += 3
            reasons.append("Fallen person/motorcycle orientation")
        elif features['aspect_ratio'] > 3.0:
            score += 1
        
        if features['brightness_variance'] > 2000:
            score += 2
            reasons.append("High contrast indicating damage")
        elif features['brightness_variance'] > 1200:
            score += 1
        
        if features['vertical_position'] > 0.5:
            score += 2
            reasons.append("Accident on main roadway")
        
        if score >= 8:
            severity = "Severe"
        elif score >= 5:
            severity = "Moderate"
        else:
            severity = "Minor"
        
        return severity, score, reasons
    
    def predict_severity_fast(self, crop_img, bbox_coords, img_shape):
        """OPTIMIZED: Ultra-fast severity prediction"""
        features = self.extract_features_fast(crop_img, bbox_coords, img_shape)
        severity, score, reasons = self.rule_based_severity_fast(features, crop_img)
        return severity, score, reasons

# ==========================================
# OPTIMIZED FIRE DETECTION
# ==========================================
class FastFireDetector:
    """Optimized fire detection"""
    
    @staticmethod
    def detect_fire_fast(frame):
        """OPTIMIZED: Ultra-fast fire detection"""
        small_frame = cv2.resize(frame, (160, 120))
        hsv = cv2.cvtColor(small_frame, cv2.COLOR_BGR2HSV)
        
        lower_fire1 = np.array([0, 100, 100])
        upper_fire1 = np.array([10, 255, 255])
        lower_fire2 = np.array([170, 100, 100])
        upper_fire2 = np.array([180, 255, 255])
        
        mask1 = cv2.inRange(hsv, lower_fire1, upper_fire1)
        mask2 = cv2.inRange(hsv, lower_fire2, upper_fire2)
        
        fire_pixels = np.count_nonzero(mask1) + np.count_nonzero(mask2)
        fire_ratio = fire_pixels / (small_frame.shape[0] * small_frame.shape[1])
        
        has_fire = fire_ratio > 0.05
        return has_fire, fire_ratio

# ==========================================
# ULTRA-FAST DETECTION PROCESSOR
# ==========================================
class UltraFastDetectionProcessor:
    """Maximum speed processor"""
    
    def __init__(self):
        try:
            self.yolo_model = YOLO(YOLO_MODEL_PATH)
            if hasattr(self.yolo_model, 'model'):
                self.yolo_model.model.eval()
            self.severity_classifier = OptimizedSeverityClassifier()
            self.fire_detector = FastFireDetector()
            self.detection_buffer = deque(maxlen=5)
            self.frame_skip_counter = 0
            self.last_fire_check = 0
            st.success("✅ Ultra-fast model loaded!")
        except Exception as e:
            st.error(f"Error loading model: {e}")
            self.yolo_model = None
    
    def process_frame_fast(self, frame, frame_count):
        """OPTIMIZED: Maximum speed processing"""
        if self.yolo_model is None:
            return [], False
        
        detections = []
        has_fire = False
        
        try:
            current_time = time.time()
            if current_time - self.last_fire_check > 0.5:
                has_fire, _ = self.fire_detector.detect_fire_fast(frame)
                self.last_fire_check = current_time
            
            processing_frame = cv2.resize(frame, (416, 416))
            
            results = self.yolo_model.predict(
                processing_frame,
                conf=0.4,
                iou=0.45,
                verbose=False,
                augment=False,
                imgsz=416,
                half=True if torch.cuda.is_available() else False
            )
            
            if len(results[0].boxes) > 0:
                scale_x = frame.shape[1] / 416
                scale_y = frame.shape[0] / 416
                
                for box in results[0].boxes:
                    confidence = float(box.conf.cpu().numpy()[0])
                    coords = box.xyxy.cpu().numpy().flatten()
                    
                    x1 = int(coords[0] * scale_x)
                    y1 = int(coords[1] * scale_y)
                    x2 = int(coords[2] * scale_x)
                    y2 = int(coords[3] * scale_y)
                    
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
                    
                    if (x2 - x1) < 20 or (y2 - y1) < 20:
                        continue
                    
                    accident_crop = frame[y1:y2, x1:x2]
                    if accident_crop.size == 0:
                        continue
                    
                    severity, score, reasons = self.severity_classifier.predict_severity_fast(
                        accident_crop,
                        (x1, y1, x2, y2),
                        frame.shape
                    )
                    
                    detection = {
                        'confidence': confidence,
                        'bbox': (x1, y1, x2, y2),
                        'severity': severity,
                        'score': score,
                        'reasons': reasons,
                        'timestamp': time.time(),
                        'has_fire': has_fire
                    }
                    
                    detections.append(detection)
                
                self.detection_buffer.append(len(detections))
            
        except Exception as e:
            st.error(f"Processing error: {e}")
        
        return detections, has_fire
    
    def get_detection_stability(self):
        """Check detection stability"""
        if len(self.detection_buffer) < 3:
            return False
        recent = list(self.detection_buffer)[-3:]
        return sum(d > 0 for d in recent) >= 2

# ==========================================
# OPTIMIZED DRAWING
# ==========================================
def draw_detection_box_fast(frame, detection):
    """OPTIMIZED: Fast drawing with minimal operations"""
    x1, y1, x2, y2 = detection['bbox']
    severity = detection['severity']
    
    colors = {'Severe': (0, 0, 255), 'Moderate': (0, 165, 255), 'Minor': (0, 255, 0)}
    color = colors.get(severity, (0, 0, 255))
    
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    
    label = f"{severity} {detection['score']}"
    if detection.get('has_fire'):
        label += " FIRE"
    
    (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(frame, (x1, y1-h-4), (x1+w, y1), color, -1)
    cv2.putText(frame, label, (x1, y1-2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return frame

# Initialize processor
@st.cache_resource
def get_processor():
    return UltraFastDetectionProcessor()

# Initialize Twilio system
@st.cache_resource
def get_twilio_system():
    return TwilioAlertSystem()

# ==========================================
# ULTRA-FAST VIDEO STREAMING
# ==========================================
def stream_video_fast(video_placeholder, metrics_placeholder, alert_placeholder):
    """OPTIMIZED: Maximum FPS streaming with Twilio alerts and UI updates"""
    processor = get_processor()
    twilio_system = get_twilio_system()
    
    twilio_system.start_alert_worker()
    
    last_metrics_update = time.time()
    last_alert_update = time.time()
    frame_times = deque(maxlen=30)
    severe_alert_sent = False
    
    while st.session_state.streaming and st.session_state.cap is not None:
        if st.session_state.stop_streaming:
            break
        
        frame_start = time.time()
        
        ret, frame = st.session_state.cap.read()
        if not ret:
            st.session_state.streaming = False
            if st.session_state.cap:
                st.session_state.cap.release()
                st.session_state.cap = None
            twilio_system.stop_alert_worker()
            st.info("✅ Video complete!")
            break
        
        st.session_state.frame_count += 1
        
        # Process frame
        detections, has_fire = processor.process_frame_fast(frame, st.session_state.frame_count)
        
        if detections:
            st.session_state.detection_count += len(detections)
            
            for det in detections:
                if det['score'] >= 5:
                    st.session_state.detections.append(det)
                    
                    # Queue Twilio alert for severe cases
                    if det['severity'] == 'Severe' and not severe_alert_sent:
                        twilio_system.queue_alert(
                            lat=st.session_state.current_lat,
                            lon=st.session_state.current_lon,
                            severity=det['severity'],
                            score=det['score'],
                            has_fire=has_fire,
                            reasons=det['reasons']
                        )
                        severe_alert_sent = True
        
        # Draw detections
        for det in detections:
            frame = draw_detection_box_fast(frame, det)
        
        # Display frame
        display_frame = cv2.resize(frame, (640, 480))
        frame_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
        video_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
        
        # Calculate FPS
        frame_time = time.time() - frame_start
        frame_times.append(frame_time)
        avg_fps = len(frame_times) / sum(frame_times) if frame_times else 0
        
        # Update metrics
        current_time = time.time()
        if current_time - last_metrics_update > 0.5:
            with metrics_placeholder.container():
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Frames", st.session_state.frame_count)
                with col2:
                    st.metric("Detections", st.session_state.detection_count)
                with col3:
                    st.metric("FPS", f"{avg_fps:.1f}")
                with col4:
                    status = "🟢 Stable" if processor.get_detection_stability() else "🟡 Tracking"
                    st.metric("Status", status)
            
            last_metrics_update = current_time
        
        # **FIX: Update alert display every 0.5 seconds**
        if current_time - last_alert_update > 0.5:
            update_alert_display(alert_placeholder)
            last_alert_update = current_time

# **FIX: New function to update alert display**
def update_alert_display(alert_placeholder):
    """Update the alert status display"""
    with alert_placeholder.container():
        st.markdown("### 🚨 Alert Status")
        
        if st.session_state.alert_sent:
            st.markdown("""
            <div class="alert-critical">
                ✅ WhatsApp Alert Sent!
            </div>
            """, unsafe_allow_html=True)
        
        # Show recent alerts
        if st.session_state.alert_status:
            st.markdown("#### Recent Alerts")
            for alert in reversed(st.session_state.alert_status[-5:]):
                if alert['status'] == 'sent':
                    status_icon = "✅ Sent"
                    css_class = "alert-success"
                elif alert['status'] == 'pending':
                    status_icon = "⏳ Sending..."
                    css_class = "alert-pending"
                else:
                    status_icon = "❌ Failed"
                    css_class = "alert-critical"
                
                fire_icon = "🔥" if alert.get('has_fire') else ""
                
                st.markdown(f"""
                <div class="{css_class}">
                    <strong>{status_icon}</strong> {alert['severity']} {fire_icon}<br>
                    <small>{alert['timestamp']}</small>
                </div>
                """, unsafe_allow_html=True)

# ==========================================
# MAIN APPLICATION
# ==========================================
def main():
    st.markdown("""
    <div class="header-container">
        <h1 style="text-align: center; color: #667eea; margin: 0;">
            🚨 Ultra-Fast Accident Detection with WhatsApp Alerts
        </h1>
        <p style="text-align: center; color: #666; margin-top: 10px;">
            Maximum FPS • Optimized Processing • Real-time Twilio Alerts
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("### ⚙ Configuration")
        
        source_type = st.radio("Source", ["Live Camera", "Upload Video"], key="source")
        
        camera_index = 0
        uploaded_file = None
        
        if source_type == "Live Camera":
            camera_index = st.number_input("Camera", 0, 5, 0)
        else:
            uploaded_file = st.file_uploader("Video", type=['mp4', 'avi', 'mov'])
        
        st.markdown("---")
        st.markdown("### 📍 Location")
        st.session_state.current_lat = st.number_input("Lat", value=DEFAULT_LAT, format="%.6f")
        st.session_state.current_lon = st.number_input("Lon", value=DEFAULT_LON, format="%.6f")
        
        st.markdown("---")
        st.markdown("### 📱 Twilio Status")
        if TWILIO_ACCOUNT_SID and EMERGENCY_NUMBER:
            st.success(f"✅ WhatsApp: {EMERGENCY_NUMBER}")
        else:
            st.error("❌ Configure .env file")
        
        st.markdown("---")
        st.markdown("### ✨ Optimizations")
        st.success("""
        **Speed Improvements:**
        - ✅ 416x416 YOLO input
        - ✅ 64x64 feature extraction
        - ✅ 160x120 fire detection
        - ✅ Async Twilio alerts
        - ✅ Minimal frame delays
        - ✅ Fast drawing (2px lines)
        - ✅ FP16 inference (GPU)
        - ✅ Non-blocking WhatsApp
        - ✅ Real-time UI updates
        """)
    
    # Main content
    st.markdown("### 📹 High-Speed Feed")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        start = st.button("🚀 Start", type="primary", use_container_width=True)
    with col2:
        stop = st.button("⏹ Stop", use_container_width=True)
    with col3:
        st.success("● Active" if st.session_state.streaming else "● Stopped")
    
    if start and not st.session_state.streaming:
        st.session_state.stop_streaming = False
        st.session_state.frame_count = 0
        st.session_state.detection_count = 0
        st.session_state.detections = []
        st.session_state.alert_sent = False
        st.session_state.alert_status = []
        
        if source_type == "Live Camera":
            cap = cv2.VideoCapture(camera_index)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            cap.set(cv2.CAP_PROP_FPS, 30)
            if cap.isOpened():
                st.session_state.cap = cap
                st.session_state.streaming = True
                st.success("✅ Camera ready!")
                st.rerun()
        elif uploaded_file:
            temp = "temp_video.mp4"
            with open(temp, 'wb') as f:
                f.write(uploaded_file.read())
            cap = cv2.VideoCapture(temp)
            if cap.isOpened():
                st.session_state.cap = cap
                st.session_state.streaming = True
                st.success("✅ Video loaded!")
                st.rerun()
    
    if stop:
        st.session_state.stop_streaming = True
        if st.session_state.cap:
            st.session_state.cap.release()
            st.session_state.cap = None
        st.session_state.streaming = False
        twilio_system = get_twilio_system()
        twilio_system.stop_alert_worker()
    
    # Metrics
    metrics_placeholder = st.empty()
    with metrics_placeholder.container():
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Frames", st.session_state.frame_count)
        with col2:
            st.metric("Detections", st.session_state.detection_count)
        with col3:
            st.metric("FPS", "0.0")
        with col4:
            st.metric("Status", "Idle")
    
    st.markdown("---")
    
    # Video and detection display
    col1, col2 = st.columns([2, 1])
    
    with col1:
        video_placeholder = st.empty()
        
        if st.session_state.streaming and st.session_state.cap:
            # **FIX: Pass alert placeholder to streaming function**
            with col2:
                alert_placeholder = st.empty()
            
            stream_video_fast(video_placeholder, metrics_placeholder, alert_placeholder)
        else:
            video_placeholder.markdown("""
            <div class="video-container">
                <p style="text-align: center; color: white; padding: 100px;">
                    🎥 Click Start for maximum FPS detection
                </p>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        # **FIX: Create persistent alert placeholder**
        if not st.session_state.streaming:
            alert_placeholder = st.empty()
            update_alert_display(alert_placeholder)
        
        st.markdown("---")
        st.markdown("### 📍 Location Map")
        
        m = folium.Map(
            location=[st.session_state.current_lat, st.session_state.current_lon],
            zoom_start=13
        )
        folium.Marker(
            [st.session_state.current_lat, st.session_state.current_lon],
            popup='<b>Accident Location</b>',
            icon=folium.Icon(color='red', icon='info-sign')
        ).add_to(m)
        st_folium(m, width=400, height=300)
    
    # Detection summary
    if st.session_state.detections:
        st.markdown("---")
        st.markdown("### 📊 Detection Summary")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            severe = sum(1 for d in st.session_state.detections if d['severity'] == 'Severe')
            st.metric("🔴 Severe", severe)
        with col2:
            moderate = sum(1 for d in st.session_state.detections if d['severity'] == 'Moderate')
            st.metric("🟠 Moderate", moderate)
        with col3:
            minor = sum(1 for d in st.session_state.detections if d['severity'] == 'Minor')
            st.metric("🟢 Minor", minor)
        with col4:
            fire = sum(1 for d in st.session_state.detections if d.get('has_fire', False))
            st.metric("🔥 Fire", fire)
        
        # Detailed detection log
        with st.expander("📋 View Detailed Detection Log"):
            for i, det in enumerate(reversed(st.session_state.detections[-10:]), 1):
                severity_emoji = {"Severe": "🔴", "Moderate": "🟠", "Minor": "🟢"}
                fire_text = " 🔥 FIRE DETECTED" if det.get('has_fire') else ""
                
                st.markdown(f"""
                **Detection #{len(st.session_state.detections) - i + 1}** {severity_emoji.get(det['severity'], '⚪')}
                - **Severity**: {det['severity']} (Score: {det['score']}/12){fire_text}
                - **Confidence**: {det['confidence']:.2%}
                - **Analysis**: {', '.join(det['reasons'][:2])}
                """)
                st.markdown("---")

if __name__ == "__main__":
    main()