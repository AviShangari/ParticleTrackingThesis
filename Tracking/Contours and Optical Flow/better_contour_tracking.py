import cv2
import numpy as np
import matplotlib.pyplot as plt
import time

# Load the video
video_path = "./Tracking/vid1.mov"
output_path = "./Tracking/Output/tracking_output.mp4"

cap = cv2.VideoCapture(video_path)

# Get video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Define Video Writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

# Read the first frame
ret, old_frame = cap.read()
if not ret:
    raise RuntimeError("Could not read the first frame.")

old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

# === Function to get all contour points (limited to 15 per contour) ===
def get_contour_points(image, max_points_per_contour=25, min_contour_area=5):
    blurred = cv2.GaussianBlur(image, (5, 5), 2)
    edges = cv2.Canny(blurred, 30, 60)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    all_points = []
    per_contour_counts = []
    for contour in contours:
        if cv2.contourArea(contour) < min_contour_area:
            continue
        contour_points = [pt[0] for pt in contour]
        sampled = contour_points[::max(1, len(contour_points) // max_points_per_contour)]
        sampled = sampled[:max_points_per_contour]
        all_points.extend(sampled)
        per_contour_counts.append(len(sampled))

    if len(all_points) == 0:
        return None, contours, []

    return np.array(all_points, dtype=np.float32).reshape(-1, 1, 2), contours, per_contour_counts

# Initialize tracking points
p0, prev_contours, prev_per_contour_counts = get_contour_points(old_gray)
if p0 is None or len(p0) == 0:
    raise ValueError("No valid contour points found for tracking!")

prev_tracked_count = len(p0)
frame_number = 1

# Optical Flow Parameters
lk_params = dict(
    winSize=(21, 21),
    maxLevel=3,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03)
)

colors = np.random.randint(0, 255, (len(p0), 3))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Optical flow tracking
    p1, status, _ = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

    if p1 is None or status is None or len(p1) == 0:
        break

    good_new = p1[status.flatten() == 1].reshape(-1, 2)
    good_old = p0[status.flatten() == 1].reshape(-1, 2)

    # Draw points
    for i, (x, y) in enumerate(good_new):
        x, y = map(int, (x, y))
        frame = cv2.circle(frame, (x, y), 3, colors[i % len(colors)].tolist(), -1)

    out.write(frame)

    # === Show frames if the number of points changes ===
    current_tracked_count = len(good_new)
    if current_tracked_count != prev_tracked_count:
        # Get contours for current frame
        _, curr_contours, curr_per_contour_counts = get_contour_points(frame_gray)

        # Draw contours separately on blank images
        prev_contour_img = np.ones_like(old_frame) * 255
        curr_contour_img = np.ones_like(frame) * 255
        cv2.drawContours(prev_contour_img, prev_contours, -1, (0, 0, 0), 1)
        for cnt in prev_contours:
            if cv2.contourArea(cnt) >= 50:
                (x, y), radius = cv2.minEnclosingCircle(cnt)
                cv2.circle(prev_contour_img, (int(x), int(y)), int(radius), (0, 0, 255), 1)
        cv2.drawContours(curr_contour_img, curr_contours, -1, (0, 0, 0), 1)
        for cnt in curr_contours:
            if cv2.contourArea(cnt) >= 50:
                (x, y), radius = cv2.minEnclosingCircle(cnt)
                cv2.circle(curr_contour_img, (int(x), int(y)), int(radius), (0, 0, 255), 1)

        # Show four images: prev frame, curr frame, prev contours, curr contours
        fig, axes = plt.subplots(2, 2, figsize=(14, 11))
        fig.suptitle(f"Frame {frame_number} | Tracked Points Changed\n"
                     f"Previous Worm Point Counts: {prev_per_contour_counts}\n"
                     f"Current Worm Point Counts: {curr_per_contour_counts}", fontsize=12)

        axes[0, 0].imshow(cv2.cvtColor(old_frame, cv2.COLOR_BGR2RGB))
        axes[0, 0].set_title("Previous Frame")
        axes[0, 0].axis("off")

        axes[0, 1].imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        axes[0, 1].set_title("Current Frame")
        axes[0, 1].axis("off")

        axes[1, 0].imshow(cv2.cvtColor(prev_contour_img, cv2.COLOR_BGR2RGB))
        axes[1, 0].set_title("Previous Contours")
        axes[1, 0].axis("off")

        axes[1, 1].imshow(cv2.cvtColor(curr_contour_img, cv2.COLOR_BGR2RGB))
        axes[1, 1].set_title("Current Contours")
        axes[1, 1].axis("off")

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()
        plt.close()

    prev_tracked_count = current_tracked_count
    old_frame = frame.copy()
    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1, 1, 2)
    _, prev_contours, prev_per_contour_counts = get_contour_points(old_gray)
    frame_number += 1

cap.release()
out.release()
print(f"Tracking video saved at: {output_path}")
