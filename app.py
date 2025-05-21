from flask import Flask, render_template, Response, redirect, url_for
import cv2
import mediapipe as mp
import numpy as np

app = Flask(__name__)

# إعداد Mediapipe
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

counter = 0
stage = None

def calculate_angle(a, b, c):
    a = np.array(a)  # الكتف
    b = np.array(b)  # الكوع
    c = np.array(c)  # المعصم

    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    
    if angle > 180.0:
        angle = 360 - angle
        
    return angle

def generate_frames():
    global counter, stage
    cap = cv2.VideoCapture(0)
    
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # عكس الصورة
            frame = cv2.flip(frame, 1)
            
            # تحويل الصورة إلى RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb_frame)
            
            # رسم العلامات
            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                
                # استخراج النقاط
                shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x * frame.shape[1],
                            landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y * frame.shape[0]]
                elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x * frame.shape[1],
                         landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y * frame.shape[0]]
                wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x * frame.shape[1],
                         landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y * frame.shape[0]]
                         
                angle = calculate_angle(shoulder, elbow, wrist)
                
                # منطق العد
                if angle > 160:
                    stage = "down"
                if angle < 45 and stage == 'down':
                    stage = "up"
                    counter += 1
                
                # عرض المعلومات على الشاشة
                cv2.putText(frame, f"العدات: {counter}", (50, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
                            
            # ترميز الصورة لـ streaming
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            
    cap.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/home')
def home():
    return render_template('home.html')

@app.route('/bicep')
def bicep():
    return render_template('bicep.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)