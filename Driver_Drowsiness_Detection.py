from scipy.spatial import distance as dist
from imutils import face_utils
import imutils
import time
import cv2
import numpy as np
from ultralytics import YOLO
import mediapipe as mp
from playsound import playsound
import winsound
import threading
from EAR import eye_aspect_ratio
print("loading YOLOv8 face detection model...")
model = YOLO("E:\CAI 2.0\DL labs\DL final project\Driver-Drowsiness-Detection\yolov8n-face.pt")
print("initializing MediaPipe Face Mesh...")
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True,min_detection_confidence=0.5, min_tracking_confidence=0.5)

print("initializing camera...")
vs = cv2.VideoCapture(0)
time.sleep(2.0)

frame_width = 1024
frame_height = 576

left_eye_indices = [362, 385, 387, 263, 373, 380] 
right_eye_indices = [33, 160, 158, 133, 153, 144] 

EYE_AR_THRESH = 0.2  
EYE_AR_CONSEC_FRAMES = 3
COUNTER = 0
was_drowsy = False 
drowsy_start_time = None  
ALERT_DURATION = 2
ALERT_REPEAT_INTERVAL = 2  

# Path to alert sound file
ALERT_SOUND = "E:\CAI 2.0\DL labs\DL final project\Driver-Drowsiness-Detection\Warning.mp3"

# Threading event to control audio playback
stop_audio_event = threading.Event()

def play_alert_sound(sound_file):
    """Play sound in a separate thread and stop if stop_audio_event is set."""
    try:
        if not stop_audio_event.is_set():
            playsound(sound_file, block=True)
    except Exception as e:
        print(f"[WARNING] Could not play sound: {e}. Falling back to winsound.")
        try:
            winsound.Beep(1000, 500)  # Fallback beep
        except Exception as e2:
            print(f"[WARNING] Winsound failed: {e2}")

while True:
    ret, frame = vs.read()
    if not ret:
        print("[ERROR] Failed to capture frame from webcam")
        break

    # Resize frame
    frame = imutils.resize(frame, width=320)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    size = gray.shape

    # YOLO face detection
    results = model(frame)
    faces = results[0].boxes.xyxy.cpu().numpy()

    # Display number of faces detected
    if len(faces) > 0:
        text = "{} face(s) found".format(len(faces))
        cv2.putText(frame, text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # Drowsiness state
    drowsy = False
    status_text = "Awake"
    status_color = (0, 255, 0)  # Green for Awake

    # Process each detected face
    for face in faces:
        x1, y1, x2, y2 = map(int, face[:4])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)

        # MediaPipe Face Mesh for landmarks
        results = face_mesh.process(rgb_frame)
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                h, w, _ = frame.shape
                landmarks = []
                for idx, lm in enumerate(face_landmarks.landmark):
                    x, y = int(lm.x * w), int(lm.y * h)
                    landmarks.append((x, y))

                # Extract eye landmarks
                leftEye = np.array([landmarks[i] for i in left_eye_indices])
                rightEye = np.array([landmarks[i] for i in right_eye_indices])

                # Compute EAR
                leftEAR = eye_aspect_ratio(leftEye)
                rightEAR = eye_aspect_ratio(rightEye)
                ear = (leftEAR + rightEAR) / 2.0

                # Draw eye contours
                leftEyeHull = cv2.convexHull(leftEye)
                rightEyeHull = cv2.convexHull(rightEye)
                cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
                cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

                # Check for eye closure
                if ear < EYE_AR_THRESH:
                    COUNTER += 1
                    if COUNTER >= EYE_AR_CONSEC_FRAMES:
                        drowsy = True
                        status_text = "Drowsy"
                        status_color = (0, 0, 255)  # Red for Drowsy
                        # Start or repeat alert sound
                        if not was_drowsy:
                            # Start new drowsy period
                            drowsy_start_time = time.time()
                            stop_audio_event.clear()  # Allow audio to play
                            threading.Thread(target=play_alert_sound, args=(ALERT_SOUND,), daemon=True).start()
                        elif time.time() - drowsy_start_time >= ALERT_REPEAT_INTERVAL:
                            # Repeat sound if drowsy for more than 10 seconds
                            stop_audio_event.clear()
                            threading.Thread(target=play_alert_sound, args=(ALERT_SOUND,), daemon=True).start()
                            drowsy_start_time = time.time()  # Reset timer
                else:
                    COUNTER = 0
                    if was_drowsy:
                        # Stop audio when switching to Awake
                        stop_audio_event.set()
                        drowsy_start_time = None

                # Update drowsiness state
                was_drowsy = drowsy

                # Display drowsiness status
                cv2.putText(frame, f"Status: {status_text}", (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

    # Show the frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

# Cleanup
stop_audio_event.set()  # Ensure audio stops on exit
cv2.destroyAllWindows()
vs.release()
face_mesh.close()
