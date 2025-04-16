import cv2
import numpy as np
import os

# === CONFIGURATION ===
VIDEO_PATH = "./Tracking/vid1.mov"
OUTPUT_DIR = "./Tracking/Output/Preprocessing_Frames"
OUTPUT_VIDEO = "./Tracking/Output/Preprocessing/preprocessing_vid_1-2.mp4"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# === Use same preprocessing logic as in your main program ===
def preprocess_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 2)
    binary = cv2.adaptiveThreshold(
        blurred, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        11, 3
    )
    return binary


# === READ AND PROCESS VIDEO ===
cap = cv2.VideoCapture(VIDEO_PATH)
frame_idx = 0
preprocessed_frames = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    binary = preprocess_frame(frame)

    # Convert binary to 3-channel grayscale for video
    binary_color = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    cv2.imwrite(os.path.join(OUTPUT_DIR, f"frame_{frame_idx:04d}.png"), binary_color)
    preprocessed_frames.append(binary_color)
    frame_idx += 1

cap.release()

# === WRITE OUTPUT VIDEO ===
if preprocessed_frames:
    height, width, _ = preprocessed_frames[0].shape
    out = cv2.VideoWriter(OUTPUT_VIDEO, cv2.VideoWriter_fourcc(*'mp4v'), 30, (width, height))

    for frame in preprocessed_frames:
        out.write(frame)
    out.release()

    print(f"Preprocessing preview video saved: {OUTPUT_VIDEO}")
else:
    print("No frames processed.")
