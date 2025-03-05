import cv2
import numpy as np

# Load the video
video_path = "./Tracking/vid1.mov"
output_path = "./Tracking/tracking_output.mp4"

cap = cv2.VideoCapture(video_path)

# Get video properties (width, height, frame rate)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Define Video Writer (codec: MP4V for MP4 files)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use 'XVID' for .avi
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

# Check if video is opened correctly
if not cap.isOpened():
    raise FileNotFoundError("Could not open video file. Check the file path and codecs.")

# Read the first frame
ret, old_frame = cap.read()
if not ret:
    raise RuntimeError("Could not read the first frame.")

# Convert first frame to grayscale
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

# Detect key points (particles) to track
feature_params = dict(maxCorners=500, qualityLevel=0.01, minDistance=3)
p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

# Ensure points exist
if p0 is None or len(p0) == 0:
    raise ValueError("No good features found to track!")

# Optical Flow Parameters
lk_params = dict(
    winSize=(21, 21),  
    maxLevel=3,        
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03)
)

# Define colors for visualization
colors = np.random.randint(0, 255, (len(p0), 3))

# Create a mask image for drawing
mask = np.zeros_like(old_frame)

while True:
    # Read the next frame
    ret, frame = cap.read()
    if not ret:
        break  # Exit loop if video ends

    # Convert to grayscale
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Track points using Optical Flow
    p1, status, _ = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

    # Check if tracking was successful
    if p1 is None or status is None or len(p1) == 0:
        break  # Exit if no points are tracked

    # Select good points
    good_new = p1[status.flatten() == 1].reshape(-1, 2)
    good_old = p0[status.flatten() == 1].reshape(-1, 2)

    # Draw tracking lines
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        x_new, y_new = map(int, new)  # Ensure integer coordinates
        x_old, y_old = map(int, old)

        mask = cv2.line(mask, (x_old, y_old), (x_new, y_new), colors[i].tolist(), 2)
        frame = cv2.circle(frame, (x_new, y_new), 3, colors[i].tolist(), -1)

    # Overlay mask on the frame
    output = cv2.add(frame, mask)

    # Write the processed frame to the output video
    out.write(output)

    # Update the previous frame and points
    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1, 1, 2)

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Tracking video saved at: {output_path}")
