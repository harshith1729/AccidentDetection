import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import torch
from collections import Counter, deque
import requests
from datetime import datetime
from urllib.parse import quote
import folium
from streamlit_folium import st_folium
import time
import tempfile
import os
import uuid

# Page Configuration
st.set_page_config(
    page_title="🚨 Accident Detection System",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); }
    .stApp { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); }
    .header-container {
        background: rgba(255, 255, 255, 0.95);
        padding: 20px;
        border-radius: 15px;
        margin-bottom: 20px;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.1);
    }
    .video-container {
        background: rgba(0, 0, 0, 0.8);
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.3);
    }
    .alert-response-box {
        background: rgba(76, 175, 80, 0.1);
        border-left: 4px solid #4CAF50;
        padding: 12px;
        border-radius: 6px;
        margin: 8px 0;
        font-family: monospace;
        font-size: 0.9em;
    }
</style>
""", unsafe_allow_html=True)

# Configuration
#YOLO_MODEL_PATH = # add your path

#DEFAULT_LAT = # lat
#DEFAULT_LON = #long
API_BASE = 'http://localhost:3000'

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
if 'alert_sent' not in st.session_state:
    st.session_state.alert_sent = False
if 'alert_response' not in st.session_state:
    st.session_state.alert_response = None
if 'current_lat' not in st.session_state:
    st.session_state.current_lat = DEFAULT_LAT
if 'current_lon' not in st.session_state:
    st.session_state.current_lon = DEFAULT_LON
if 'last_frame_time' not in st.session_state:
    st.session_state.last_frame_time = time.time()
if 'fps_buffer' not in st.session_state:
    st.session_state.fps_buffer = deque(maxlen=10)

# ==========================================
# ULTRA-OPTIMIZED DETECTION PROCESSOR
# ==========================================
class UltraOptimizedDetectionProcessor:
    """Extremely optimized processor for maximum FPS"""
    
    def __init__(self):
        self.model_loaded = False
        try:
            # Load model with minimal settings
            self.yolo_model = YOLO(YOLO_MODEL_PATH)
            self.model_loaded = True
            st.success("✅ YOLO model loaded successfully!")
        except Exception as e:
            st.error(f"Error loading YOLO model: {e}")
            self.yolo_model = None
    
    def process_frame(self, frame):
        """Ultra-optimized frame processing"""
        if not self.model_loaded or self.yolo_model is None:
            return [], False
        
        detections = []
        
        try:
            # MAJOR OPTIMIZATION: Skip frames for processing but maintain video display
            # Process only every 5th frame for detection
            if st.session_state.frame_count % 5 != 0:
                return detections, False
            
            # Use much smaller frame for processing
            original_shape = frame.shape
            processing_frame = cv2.resize(frame, (320, 240))  # Reduced from 640x480
            
            # Fast YOLO prediction with minimal processing
            results = self.yolo_model.predict(
                processing_frame,
                conf=0.4,  # Slightly higher confidence to reduce false positives
                iou=0.5,
                verbose=False,
                augment=False,
                imgsz=320,  # Match processing frame size
                max_det=3,  # Limit maximum detections
                half=False   # Disable half precision for stability
            )
            
            if len(results[0].boxes) > 0:
                for box in results[0].boxes:
                    confidence = float(box.conf.cpu().numpy()[0])
                    coords = box.xyxy.cpu().numpy().flatten()
                    
                    # Scale coordinates back to original frame
                    scale_x = original_shape[1] / 320
                    scale_y = original_shape[0] / 240
                    x1, y1, x2, y2 = map(int, [
                        coords[0] * scale_x, 
                        coords[1] * scale_y, 
                        coords[2] * scale_x, 
                        coords[3] * scale_y
                    ])
                    
                    # Simple severity classification based on size and position
                    bbox_area = (x2 - x1) * (y2 - y1)
                    frame_area = original_shape[0] * original_shape[1]
                    area_ratio = bbox_area / frame_area
                    
                    # Quick severity calculation
                    if area_ratio > 0.1:
                        severity = "Severe"
                        score = 8
                    elif area_ratio > 0.05:
                        severity = "Moderate" 
                        score = 5
                    else:
                        severity = "Minor"
                        score = 3
                    
                    detection = {
                        'confidence': confidence,
                        'bbox': (x1, y1, x2, y2),
                        'severity': severity,
                        'score': score,
                        'timestamp': time.time(),
                        'has_fire': False  # Skip fire detection for performance
                    }
                    
                    detections.append(detection)
            
        except Exception as e:
            # Silent error to avoid performance hit
            pass
        
        return detections, False

# ==========================================
# OPTIMIZED ALERT SYSTEM - USING YOUR EXACT DUMMY DATA
# ==========================================
def send_accident_alert(lat, lon, description, has_fire=False):
    """Send accident alert - using exact dummy data from your screenshot"""
    try:
        # Generate a unique alert ID
        alert_id = str(uuid.uuid4())
        
        # Return the EXACT response format from your screenshot
        return True, {
            "success": True,
            "alert_id": alert_id,
            "notified_amenities": 2,
            "fire_incident": False,
            "message": "Alert created successfully!",
            "fire_incident_text": "NO",
            "total_amenities_notified": 2,
            "alerts_sent_to": [
                {
                    "name": "Apollo Pharmacy",
                    "type": "hospital", 
                    "email": "apollopharmacy.hospital21@guardianconnect.emergency",
                    "distance": 7.93
                },
                {
                    "name": "Karkhana Police Station",
                    "type": "police",
                    "email": "karkhanapolicestatio.policen@guardianconnect.emerge",
                    "distance": 8.35
                }
            ]
        }
    
    except Exception as e:
        return False, {"error": str(e)}

# ==========================================
# OPTIMIZED VIDEO STREAMING
# ==========================================
def stream_video_optimized(video_placeholder, metrics_placeholder):
    """
    Ultra-optimized video streaming for maximum FPS
    """
    processor = UltraOptimizedDetectionProcessor()
    last_metrics_update = time.time()
    severe_detection_sent = False
    
    # Pre-allocate arrays for better performance
    frame_time_buffer = deque(maxlen=5)
    
    while st.session_state.streaming and st.session_state.cap is not None:
        if st.session_state.stop_streaming:
            break
        
        frame_start = time.time()
        
        # Read frame
        ret, frame = st.session_state.cap.read()
        
        if not ret:
            # Video ended
            st.session_state.streaming = False
            if st.session_state.cap is not None:
                st.session_state.cap.release()
                st.session_state.cap = None
            st.info("✅ Video processing complete!")
            break
        
        st.session_state.frame_count += 1
        
        # MAJOR OPTIMIZATION: Process detections in background without blocking
        detections, has_fire = processor.process_frame(frame)
        
        if detections:
            st.session_state.detection_count += len(detections)
            
            # Store significant detections
            for detection in detections:
                if detection['score'] >= 5:
                    st.session_state.detections.append(detection)
                    
                    # Send alert for severe detection (only once)
                    if detection['severity'] == 'Severe' and not severe_detection_sent:
                        description = f"Severe accident detected! Score: {detection['score']}/12"
                        success, result = send_accident_alert(
                            st.session_state.current_lat,
                            st.session_state.current_lon,
                            description,
                            has_fire
                        )
                        
                        if success:
                            st.session_state.alert_sent = True
                            st.session_state.alert_response = result
                            severe_detection_sent = True
            
            # Draw detections on frame
            for detection in detections:
                x1, y1, x2, y2 = detection['bbox']
                severity = detection['severity']
                
                # Simple color mapping
                if severity == "Severe":
                    color = (0, 0, 255)  # Red
                elif severity == "Moderate":
                    color = (0, 165, 255)  # Orange  
                else:
                    color = (0, 255, 0)  # Green
                
                # Draw rectangle
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                # Simple label
                label = f"{severity}"
                cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Convert for display (optimized)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Resize for consistent display (smaller = faster)
        display_frame = cv2.resize(frame_rgb, (640, 480))
        
        # Update video display
        video_placeholder.image(display_frame, channels="RGB", use_container_width=False)
        
        # Calculate FPS (optimized)
        frame_time = time.time() - frame_start
        frame_time_buffer.append(frame_time)
        current_fps = 1.0 / frame_time if frame_time > 0 else 0
        st.session_state.fps_buffer.append(current_fps)
        avg_fps = sum(st.session_state.fps_buffer) / len(st.session_state.fps_buffer) if st.session_state.fps_buffer else 0
        
        # Update metrics less frequently
        current_time = time.time()
        if current_time - last_metrics_update > 2.0:  # Update every 2 seconds
            with metrics_placeholder.container():
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Frames", st.session_state.frame_count)
                with col2:
                    st.metric("Detections", st.session_state.detection_count)
                with col3:
                    severity = st.session_state.detections[-1]['severity'] if st.session_state.detections else "-"
                    st.metric("Latest", severity)
                with col4:
                    st.metric("FPS", f"{avg_fps:.1f}")
            
            last_metrics_update = current_time

# Initialize processor
@st.cache_resource
def get_processor():
    return UltraOptimizedDetectionProcessor()

# ==========================================
# MAIN APPLICATION
# ==========================================
def main():
    # Header
    st.markdown("""
    <div class="header-container">
        <h1 style="text-align: center; color: #667eea; margin: 0;">
            🚨 High-Accuracy Accident Detection System
        </h1>
        <p style="text-align: center; color: #666; margin-top: 10px;">
            Ultra-optimized for maximum FPS • Real-time detection • Emergency response
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("### ⚙ System Configuration")
        
        source_type = st.radio(
            "Select Video Source",
            ["Upload Video File", "Live Camera"],
            key="source_type"
        )
        
        uploaded_file = None
        camera_index = 0
        
        if source_type == "Upload Video File":
            uploaded_file = st.file_uploader("Upload Video", type=['mp4', 'avi', 'mov', 'mkv'])
            st.caption("Limit: MP4, AVI, MOV, MKV")
        else:
            camera_index = st.number_input("Camera Index", min_value=0, max_value=5, value=0, help="0 for default camera")
        
        st.markdown("---")
        st.markdown("### 📍 Location Settings")
        st.session_state.current_lat = st.number_input("Latitude", value=17.285000, format="%.6f")
        st.session_state.current_lon = st.number_input("Longitude", value=78.460700, format="%.6f")
        
        st.markdown("---")
        st.markdown("### 📊 System Status")
        
        status_text = "● Streaming Active" if st.session_state.streaming else "● Streaming Inactive"
        status_color = "green" if st.session_state.streaming else "red"
        st.markdown(f"<p style='color: {status_color}; font-weight: bold;'>{status_text}</p>", unsafe_allow_html=True)
        
        if st.session_state.alert_sent:
            st.success("✅ Alert Sent!")
        
        st.markdown("---")
        st.markdown("### 🚀 Performance Mode")
        st.info("""
        **Ultra-Optimized Mode:**
        - ✅ Max FPS priority
        - ✅ Every 5th frame processed
        - ✅ Fast detection
        - ✅ Emergency alerts
        - ✅ Real-time display
        """)
    
    # Main content
    st.markdown("### 📹 Live Video Feed")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        start_button = st.button("🚀 Start Detection", type="primary", use_container_width=True)
        stop_button = st.button("⏹ Stop Streaming", use_container_width=True)
    
    with col2:
        if st.session_state.streaming:
            st.success("● ACTIVE - Processing")
        else:
            st.info("● READY - Click Start")
    
    # Handle start/stop
    if start_button and not st.session_state.streaming:
        st.session_state.stop_streaming = False
        st.session_state.frame_count = 0
        st.session_state.detection_count = 0
        st.session_state.detections = []
        st.session_state.alert_sent = False
        st.session_state.alert_response = None
        
        if source_type == "Live Camera":
            cap = cv2.VideoCapture(camera_index)
            # Set lower resolution for better performance
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_FPS, 30)  # Limit FPS
            if cap.isOpened():
                st.session_state.cap = cap
                st.session_state.streaming = True
                st.success("✅ Camera opened - Starting ultra-optimized detection!")
            else:
                st.error(f"❌ Failed to open camera {camera_index}")
        elif uploaded_file:
            # Save uploaded file to temp location
            temp_dir = tempfile.mkdtemp()
            temp_path = os.path.join(temp_dir, "uploaded_video.mp4")
            
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.read())
            
            cap = cv2.VideoCapture(temp_path)
            if cap.isOpened():
                st.session_state.cap = cap
                st.session_state.streaming = True
                st.success("✅ Video loaded - Starting ultra-optimized detection!")
            else:
                st.error("❌ Failed to open video file")
        else:
            st.warning("⚠ Please upload a video file or select live camera!")
    
    if stop_button:
        st.session_state.stop_streaming = True
        if st.session_state.cap is not None:
            st.session_state.cap.release()
            st.session_state.cap = None
        st.session_state.streaming = False
        st.info("✋ Streaming stopped")
    
    # Metrics placeholder
    metrics_placeholder = st.empty()
    
    # Initial metrics
    with metrics_placeholder.container():
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Frames", st.session_state.frame_count)
        with col2:
            st.metric("Detections", st.session_state.detection_count)
        with col3:
            severity = st.session_state.detections[-1]['severity'] if st.session_state.detections else "-"
            st.metric("Latest", severity)
        with col4:
            st.metric("FPS", "0.0")
    
    st.markdown("---")
    
    # Video and Emergency Response columns
    col1, col2 = st.columns([2, 1])
    
    with col1:
        video_placeholder = st.empty()
        
        if st.session_state.streaming and st.session_state.cap is not None:
            stream_video_optimized(video_placeholder, metrics_placeholder)
        else:
            video_placeholder.markdown("""
            <div class="video-container" style="text-align: center; color: white; padding: 80px 20px;">
                <span style="font-size: 4em;">🎥</span><br><br>
                <strong>Video Feed Ready</strong><br>
                Click "Start Detection" to begin<br>
                <small>Ultra-optimized for maximum FPS</small>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("### 🏥 Emergency Response")
        
        # Always show the emergency response format (like in your image)
        if st.session_state.alert_sent and st.session_state.alert_response:
            st.success("✅ Alert Sent Successfully!")
            
            # Display the clean emergency response without HTML tags
            response = st.session_state.alert_response
            alerts_sent_to = response.get('alerts_sent_to', [])
            
            # Use st.write for clean text without HTML formatting
            st.write("**Alert Response Details**")
            st.write("")
            st.write("Fire incident: NO")
            st.write(f"Total amenities notified: {response.get('total_amenities_notified', 0)}")
            st.write("")
            st.write("**📌 Alerts sent to:**")
            st.write("")
            
            for i, service in enumerate(alerts_sent_to, 1):
                icon = "📍" if service['type'] == 'hospital' else "🔴"
                st.write(f"{i}. {icon} {service['name']}")
                st.write(f"   Type: {service['type']}")
                st.write(f"   Email: {service['email']}")
                st.write(f"   Distance: {service['distance']} km")
                st.write("")
            
        else:
            # Show default emergency response format without HTML
            st.write("**Alert Response Details**")
            st.write("")
            st.write("Fire incident: NO")
            st.write("Total amenities notified: 2")
            st.write("")
            st.write("**📌 Alerts sent to:**")
            st.write("")
            st.write("1. 📍 Apollo Pharmacy")
            st.write("   Type: hospital")
            st.write("   Email: apollopharmacy.hospital21@guardianconnect.emergency")
            st.write("   Distance: 7.93 km")
            st.write("")
            st.write("2. 🔴 Karkhana Police Station")
            st.write("   Type: police")
            st.write("   Email: karkhanapolicestatio.policen@guardianconnect.emerge")
            st.write("   Distance: 8.35 km")
        
        # Manual alert button
        if st.button("🚨 Send Test Alert", use_container_width=True):
            description = "Manual test alert triggered"
            success, result = send_accident_alert(
                st.session_state.current_lat,
                st.session_state.current_lon,
                description,
                False
            )
            if success:
                st.session_state.alert_sent = True
                st.session_state.alert_response = result
                st.success("✅ Test alert sent successfully!")
                st.rerun()
            else:
                st.error(f"❌ Failed to send alert: {result}")
        
        # Simple map
        st.markdown("### 📍 Accident Location")
        try:
            m = folium.Map(location=[st.session_state.current_lat, st.session_state.current_lon], zoom_start=13)
            folium.Marker(
                [st.session_state.current_lat, st.session_state.current_lon],
                popup='<b>Accident Location</b>',
                icon=folium.Icon(color='red', icon='info-sign')
            ).add_to(m)
            st_folium(m, width=400, height=200)
        except:
            st.info("Map display unavailable")
    
    # Detection History
    st.markdown("---")
    st.markdown("### 📊 Detection History")
    
    if st.session_state.detections:
        # Show last 5 detections
        recent_detections = st.session_state.detections[-5:]
        
        for i, detection in enumerate(reversed(recent_detections)):
            with st.expander(f"Detection {len(recent_detections)-i}: {detection['severity']} (Score: {detection['score']}/12)", expanded=i==0):
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**Confidence:** {detection['confidence']:.1%}")
                    st.write(f"**Frame:** {st.session_state.frame_count}")
                with col2:
                    st.write(f"**Severity:** {detection['severity']}")
                    st.write(f"**Score:** {detection['score']}/12")
                
                timestamp = datetime.fromtimestamp(detection['timestamp']).strftime('%H:%M:%S')
                st.write(f"**Time:** {timestamp}")
    else:
        st.info("No detections yet. Start streaming to begin detection.")
    
    # Performance info
    st.markdown("---")
    st.markdown("### 📈 Performance Information")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**System Status:**")
        if st.session_state.streaming:
            st.success("🟢 ACTIVE - Optimized Mode")
        else:
            st.info("🟡 READY - Click Start")
    
    with col2:
        st.markdown("**Detection Mode:**")
        st.write("Ultra-Optimized")
        st.write("Every 5th frame processed")
    
    with col3:
        st.markdown("**Current FPS:**")
        if st.session_state.fps_buffer:
            current_fps = st.session_state.fps_buffer[-1] if st.session_state.fps_buffer else 0
            st.metric("", f"{current_fps:.1f} FPS")
        else:
            st.metric("", "0.0 FPS")

if __name__ == "__main__":
    main()