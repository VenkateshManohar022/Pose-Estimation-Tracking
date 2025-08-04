import numpy as np
import cv2

def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba, bc = a - b, c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)

#Implemented Rula Score here as a Protoype which is a Base Implementation.
def get_rula_score(angle):
    if angle <= 20:
        return 1
    elif angle <= 45:
        return 2
    elif angle <= 90:
        return 3
    else:
        return 4

def get_risk_label(rula_score):
    if rula_score <= 2:
        return "Low"
    elif rula_score <= 4:
        return "Medium"
    else:
        return "High"
    

# --- Fatigue prediction logic ---
def predict_fatigue(rula_history, fps=30, window_seconds=5, threshold=5):
    max_len = int(fps * window_seconds)
    recent_scores = rula_history[-max_len:]
    avg_rula = sum(recent_scores) / len(recent_scores) if recent_scores else 0

    if avg_rula >= threshold:
        return "HIGH-RISK", (0,   0, 255)
    elif avg_rula >= threshold - 1:
        return "MEDIUM",    (0, 165, 255)
    else:
        return "LOW",       (0, 255,   0)