import cv2
import numpy as np
from ultralytics import YOLO
import functions as func
import os

model = YOLO("models/yolo11l-pose.pt")

video_name='demo4.mp4'
video_path = f"videos/{video_name}"
# video_path = "/Users/manohar/Documents/GCP_ML_Engineer/Code/Lab/11_usecase/videos/demo2.mp4"
cap = cv2.VideoCapture(video_path)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Efficient codec
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
os.makedirs('runs/outputs', exist_ok=True)
out = cv2.VideoWriter(f"runs/outputs/{video_name}", fourcc, fps, (width, height))
person_history = {}

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Empty gray heatmap for this frame
    heatmap_gray = np.zeros((height, width), dtype=np.uint8)

    results = model.track(frame, persist=True, conf=0.3, save=False)

    annotated_frame = frame.copy()   # draw posture info here

    if results[0].keypoints is not None and results[0].boxes.id is not None:
        track_ids = results[0].boxes.id.int().cpu().numpy()
        keypoints = results[0].keypoints.data.cpu().numpy()
        boxes = results[0].boxes.xyxy.int().cpu().numpy()

        for (kpts, track_id, box) in zip(keypoints, track_ids, boxes):

            LEFT_SHOULDER = kpts[5][:2]
            LEFT_ELBOW = kpts[7][:2]
            LEFT_WRIST = kpts[9][:2]

            if np.all(LEFT_SHOULDER) and np.all(LEFT_ELBOW) and np.all(LEFT_WRIST):
                angle = func.calculate_angle(LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST)
                rula_score = func.get_rula_score(angle)
                risk = func.get_risk_label(rula_score)

                #Fatigue Risk
                if track_id not in person_history:
                    person_history[track_id] = []
                person_history[track_id].append(rula_score)

                fatigue_status, fat_color = func.predict_fatigue(person_history[track_id],
                                            fps=fps, window_seconds=5,
                                            threshold=5)


                # ---------------- Custom Heat Intensity ---------------- #
                # Normalize RULA (1-7) → 0-255
                heat_value = int(((rula_score - 1) / 6) * 255)

                # draw rectangle on grayscale heatmap
                x1, y1, x2, y2 = box
                cv2.rectangle(heatmap_gray, (x1, y1), (x2, y2), heat_value, thickness=-1)

                # Draw the background and text on the 'annotated_frame'
                cv2.rectangle(annotated_frame, 
                      (x1, y1 - 60), (x2, y1),
                      (20, 20, 20), -1)
        
                # Display RULA Score and Risk
                cv2.putText(annotated_frame, 
                    f"ID: {track_id} | RULA: {rula_score} ", 
                    (x1, y1 - 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                # Display Fatigue Status
                cv2.putText(annotated_frame, 
                    f"Fatigue: {fatigue_status}", 
                    (x1, y1 - 15), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, fat_color, 2)
                    

    # Threshold/mask the background
    mask = heatmap_gray > 0        # True where value > 0 (i.e., foreground)

    # Apply colormap as usual
    colored = cv2.applyColorMap(heatmap_gray, cv2.COLORMAP_JET)    # or any map

    # Make background black (0) in the colored image
    colored[~mask] = (0, 0, 0)
    
 
    alpha = 0.6
    combined = annotated_frame.copy()
    # combined = cv2.addWeighted(annotated_frame, alpha, annotated_frame, 1 - alpha, 0)
    # Only apply blending where heatmap has value
    if np.any(mask):   # only blend if there are hotspot pixels
        blended_pixels = cv2.addWeighted(colored[mask].astype(np.uint8), alpha,
                                     annotated_frame[mask].astype(np.uint8),
                                     1 - alpha, 0)
        combined[mask] = blended_pixels

    # cv2.imshow("Custom Heatmap on RULA", combined)
    out.write(combined)

    # delay=int(1000/fps)

    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break

cap.release()
out.release()
cv2.destroyAllWindows()