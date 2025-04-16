import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load video
video_path = "./Tracking/vid1.mov"
cap = cv2.VideoCapture(video_path)

# Get video properties
fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Store tracking data
worm_tracks = []  
worm_speeds = []  

# Read first frame
ret, old_frame = cap.read()
if not ret:
    raise RuntimeError("Could not read the first frame.")

old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

def get_contour_points(image):
    """Extracts key points from contours."""
    blurred = cv2.GaussianBlur(image, (5, 5), 0)  
    edges = cv2.Canny(blurred, 20, 55)  
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    points = []
    for contour in contours:
        for pt in contour:
            points.append(pt[0])  

    if len(points) == 0:
        return None

    return np.array(points, dtype=np.float32).reshape(-1, 1, 2)

# Get initial contour points
p0 = get_contour_points(old_gray)
if p0 is None or len(p0) == 0:
    raise ValueError("No valid contour points found for tracking!")

# Optical Flow Parameters
lk_params = dict(
    winSize=(21, 21),  
    maxLevel=3,        
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03)
)

# Initialize previous positions
prev_positions = p0.reshape(-1, 2)
frame_duration = 1 / fps  # Time per frame in seconds

# Process video frames
frame_count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break  

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Track points using Optical Flow
    p1, status, _ = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

    if p1 is None or status is None or len(p1) == 0:
        break  

    # Select only successfully tracked points
    valid_idx = status.flatten() == 1  
    good_new = p1[valid_idx].reshape(-1, 2)
    good_old = prev_positions[valid_idx].reshape(-1, 2)

    if len(good_new) == 0:
        break 

    # Calculate speed: Speed = Distance / Time
    speeds = np.linalg.norm(good_new - good_old, axis=1) / frame_duration  
    avg_speed = np.mean(speeds) if len(speeds) > 0 else 0
    worm_speeds.append(avg_speed)

    # Store trajectory
    worm_tracks.append(good_new)

    # Update previous frame and points
    old_gray = frame_gray.copy()
    prev_positions = good_new.copy()
    p0 = good_new.reshape(-1, 1, 2)

    frame_count += 1


cap.release()

# Convert lists to NumPy arrays for analysis
worm_speeds = np.array(worm_speeds)

# Calculate movement statistics
average_speed = np.mean(worm_speeds)  

print(f"Movement Analysis Results:")
print(f"Average Speed of Worms: {average_speed:.2f} pixels/sec")

# Scatterplot of Speed Over Time
plt.figure(figsize=(10, 5))
plt.scatter(range(len(worm_speeds)), worm_speeds, color='red', alpha=0.6, label="Speed per Frame")
plt.xlabel("Frame Number")
plt.ylabel("Speed (pixels/sec)")
plt.title("Worm Speed Scatter Plot Over Time")
plt.legend()
plt.show()