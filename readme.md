# Detect Ergonomic Risk from Operator Posture using RULA Scoring

## 🧠 Problem

In industrial and manufacturing environments, poor posture can lead to serious musculoskeletal disorders (MSDs) and long-term fatigue. Traditional ergonomic assessments like RULA (Rapid Upper Limb Assessment) are manual, time-consuming, and prone to human error. There’s a need for an **automated system that can assess posture in real-time** and highlight ergonomic risks early.

## 📊 Data

The input to this project is:
- **Video or image streams** of workers performing tasks.
- **Pose keypoints** extracted using computer vision models like YOLO-Pose, MediaPipe, or OpenPose.

Each frame of the video is processed to detect human joints such as:
- Neck, Shoulder, Elbow, Wrist
- Hip, Knee, Ankle

These keypoints are used to calculate joint angles.

> 💡 No external dataset is used; pose estimation runs on raw video/images in real-time.

## 🧮 Model & Logic

This project doesn't train a machine learning model. Instead, it applies domain knowledge to calculate RULA scores.

### Pipeline Steps:
1. **Pose Estimation**  
   Using pre-trained models (e.g., YOLO-Pose), extract 2D keypoints for each frame.

2. **Angle Calculation**  
   Compute joint angles (e.g., neck-shoulder-elbow, hip-knee-ankle) using vector geometry.

3. **RULA Scoring**  
   Apply RULA tables to convert joint angles to risk scores.  
   RULA scores range from **1 (low risk)** to **7 (high risk)**.

4. **Fatigue Tracking (Bonus)**  
   Optionally, track ergonomic risk scores over time to assess cumulative fatigue.

5. **Output**  
   Display:
   - Annotated video with real-time RULA score
   - Heatmap overlay showing high-risk zones on the body

## 📈 Evaluation Metrics

Since this is a rule-based system, traditional ML metrics (accuracy, precision) do not apply. Instead, we evaluate:

| Metric | Description |
|--------|-------------|
| **Score Accuracy** | Whether calculated scores match manual RULA assessment |
| **Frame Rate (FPS)** | Real-time performance on video feeds |
| **Latency** | Time to process each frame |
| **Consistency** | Score stability over different videos and postures |

For validation, results can be compared with expert-assessed RULA scores on selected sample frames.

## 🌍 Real-World Use Case

This solution can be deployed in:

### 🏭 Factories and Assembly Lines
- Monitor workers in real-time.
- Trigger alerts if posture becomes risky.
- Use long-term scoring for fatigue management.

### 🧑‍⚕️ Occupational Health and Safety Departments
- Replace manual ergonomic audits with automated video-based analysis.
- Create dashboards showing average risk across shifts.

### 💻 Remote Work / Office Setup
- Apply similar logic to desk workers using webcams.
- Suggest ergonomic improvements based on posture detection.

### 🚀 Future Integration
- Link with IoT wearables for angle confirmation.
- Integrate into HR compliance tools for regular posture audits.

## 📦 Folder Structure
├── models/                 # Pose estimation models (e.g., YOLO11l-pose .pt files)
├── inputs_Git/                 # Input videos
├── outputs_Git/            # Output annotated videos
├── functions.py            # Angle calculation and RULA logic
├── main.py                 # Main script to run the pipeline
├── README.md               # Project overview (this file)


## 🛠️ Technologies Used

- Python
- OpenCV (for Processing, Boxes, heatmap overlays)
- Ultralytics YOLO-Pose / MediaPipe / OpenPose
- NumPy
- Object-Tracking Technique

## ✅ Setup Instructions

```bash
git clone https://github.com/VenkateshManohar022/Pose-Estimation-Tracking.git
cd Pose-Estimation-Tracking
pip install -r requirements.txt
python main.py --video videos/input.mp4