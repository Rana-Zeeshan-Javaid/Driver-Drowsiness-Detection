# 🚗 Driver Drowsiness Detection using MediaPipe and YOLOv8

A real-time **driver drowsiness detection system** built with **YOLOv8**, **MediaPipe Face Mesh**, and **OpenCV**.  
It detects when the driver's eyes remain closed for a certain duration and triggers an audio alert to prevent accidents.

---

## 🧠 Features

✅ Real-time face detection using **YOLOv8**  
✅ Eye landmark tracking using **MediaPipe Face Mesh**  
✅ **Eye Aspect Ratio (EAR)** calculation for detecting eye closure  
✅ Sound alert system using `playsound` and threading  
✅ Lightweight and runs on any standard webcam  
✅ Extensible for head tilt and yawning detection

---

## 🧩 Tech Stack

- **Python 3.10+**
- **YOLOv8 (Ultralytics)**
- **MediaPipe**
- **OpenCV**
- **SciPy**
- **imutils**
- **playsound / winsound**

---

## ⚙️ How It Works

1. The webcam captures real-time frames.  
2. YOLOv8 detects the face region.  
3. MediaPipe Face Mesh extracts facial landmarks.  
4. Eye Aspect Ratio (EAR) is calculated using key eye points.  
5. If EAR < 0.2 for several consecutive frames → driver is **drowsy**.  
6. An alert sound is played until eyes reopen.

---

## 🧮 Eye Aspect Ratio (EAR)

\[
EAR = \frac{||p2 - p6|| + ||p3 - p5||}{2 \times ||p1 - p4||}
\]

This ratio drops when eyes are closed, helping detect drowsiness accurately.

---

## 🚨 Example Output

- **Awake:** Green text and bounding box  
- **Drowsy:** Red text and an alert sound plays  

*(You can add a screenshot or demo GIF here)*

---

## 💻 Run It Yourself

```bash
# Clone the repository
git clone https://github.com/yourusername/Driver-Drowsiness-Detection.git

# Navigate to the folder
cd Driver-Drowsiness-Detection

# Install dependencies
pip install -r requirements.txt

# Run the main script
python main.py
