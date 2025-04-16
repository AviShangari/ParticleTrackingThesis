import cv2
import numpy as np
from skimage.morphology import skeletonize
from skimage.graph import route_through_array
from skimage.util import invert
from scipy.ndimage import convolve
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt
import os

#  CONFIGURATION 
VIDEO_PATH = "./Tracking/vid1.mov"
OUTPUT_DIR = "./Tracking/Output/worm_tracks"
KEYPOINTS_PER_WORM = 15
AREA_THRESHOLD = 50

os.makedirs(OUTPUT_DIR, exist_ok=True)

#  UTILITIES 
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

def extract_worm_masks(binary):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary)
    masks = []
    height = binary.shape[0]
    for i in range(1, num_labels):  # skip background
        if stats[i, cv2.CC_STAT_AREA] >= AREA_THRESHOLD:
            y = stats[i, cv2.CC_STAT_TOP]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            # Skip if the bounding box touches the bottom edge
            if y + h >= height - 5:
                continue
            mask = (labels == i).astype(np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
            masks.append(mask)
    return masks

# Additional imports used in skeleton function
from skimage.graph import route_through_array
from skimage.util import invert
from scipy.ndimage import convolve


def get_skeleton_points(mask, num_points):
    # Slightly dilate the mask before skeletonizing to ensure continuity
    mask_dilated = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1)
    skeleton = skeletonize(mask_dilated > 0)

    if np.count_nonzero(skeleton) < 2:
        return None

    # Get endpoints = skeleton pixels with only one neighbor
    kernel = np.ones((3, 3), dtype=int)
    neighbor_count = convolve(skeleton.astype(int), kernel, mode='constant') * skeleton
    endpoints = np.column_stack(np.where(neighbor_count == 2))

    if len(endpoints) < 2:
        return None

    # Route through skeleton from endpoint to endpoint
    start = tuple(endpoints[0])
    end = tuple(endpoints[-1])
    try:
        path, _ = route_through_array(invert(skeleton).astype(np.float32), start, end, fully_connected=True)
    except Exception:
        return None

    path = np.array(path)
    if len(path) < 2:
        return None

    # Interpolate fixed number of points along the path
    indices = np.linspace(0, len(path) - 1, num=num_points, dtype=int)
    return path[indices]

def compute_cost_matrix(current_pts, prev_pts):
    cost = np.zeros((len(prev_pts), len(current_pts)))
    for i, prev in enumerate(prev_pts):
        for j, curr in enumerate(current_pts):
            centroid_dist = np.linalg.norm(np.mean(prev, axis=0) - np.mean(curr, axis=0))
            shape_dist = np.mean(np.linalg.norm(prev - curr, axis=1))
            cost[i, j] = 0.4 * centroid_dist + 0.6 * shape_dist
    return cost

def draw_tracks(frame, worm_keypoints, worm_ids):
    static_colors = {}
    for i, points in enumerate(worm_keypoints):
        worm_id = worm_ids[i]
        if worm_id not in static_colors:
            np.random.seed(worm_id)
            static_colors[worm_id] = tuple(int(x) for x in np.random.randint(100, 255, 3))
        color = static_colors[worm_id]
        for pt in points:
            x, y = int(pt[1]), int(pt[0])
            cv2.circle(frame, (x, y), 4, color, -1)
        for j in range(1, len(points)):
            pt1 = (int(points[j-1][1]), int(points[j-1][0]))
            pt2 = (int(points[j][1]), int(points[j][0]))
            cv2.line(frame, pt1, pt2, color, 2)
        cv2.putText(frame, f"ID {worm_id}", tuple(points[0][::-1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return frame

#  MAIN PIPELINE 
cap = cv2.VideoCapture(VIDEO_PATH)
frame_idx = 0
track_memory = []  # list of dicts: {id, keypoints, age}
next_id = 0
MAX_AGE = 5
next_id = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    binary = preprocess_frame(frame)
    masks = extract_worm_masks(binary)
    current_keypoints = []

    for mask in masks:
        keypoints = get_skeleton_points(mask, KEYPOINTS_PER_WORM)
        if keypoints is not None:
            current_keypoints.append(keypoints)

    # Match current worms to previous using centroid distance
    current_ids = []
    if len(track_memory) == 0:
        current_ids = list(range(next_id, next_id + len(current_keypoints)))
        next_id += len(current_keypoints)
    else:
        prev_keypoints = [t["keypoints"] for t in track_memory if t["age"] <= MAX_AGE]
        prev_ids = [t["id"] for t in track_memory if t["age"] <= MAX_AGE]
        cost = compute_cost_matrix(current_keypoints, prev_keypoints)
        row_ind, col_ind = linear_sum_assignment(cost)

        matched = set()
        current_ids = [-1] * len(current_keypoints)
        for i, j in zip(row_ind, col_ind):
            if i < len(prev_keypoints) and j < len(current_keypoints):
                cost_val = cost[i, j]
                if cost_val < 80:
                    current_ids[j] = prev_ids[i]
                    matched.add(j)

        for j in range(len(current_keypoints)):
            if current_ids[j] == -1:
                curr_centroid = np.mean(current_keypoints[j], axis=0)
                best_dist = float('inf')
                best_match = -1
                for i in range(len(prev_keypoints)):
                    if prev_ids[i] not in current_ids:
                        prev_centroid = np.mean(prev_keypoints[i], axis=0)
                        dist = np.linalg.norm(curr_centroid - prev_centroid)
                        if dist < best_dist and dist < 100:
                            best_dist = dist
                            best_match = i
                if best_match != -1:
                    current_ids[j] = prev_ids[best_match]
                else:
                    current_ids[j] = next_id
                    next_id += 1

        # Update track memory
    updated_tracks = []
    for tid, kps in zip(current_ids, current_keypoints):
        updated_tracks.append({"id": tid, "keypoints": kps, "age": 0})

    for old_track in track_memory:
        if old_track["id"] not in current_ids and old_track["age"] < MAX_AGE:
            updated_tracks.append({"id": old_track["id"], "keypoints": old_track["keypoints"], "age": old_track["age"] + 1})

    track_memory = updated_tracks

    # Draw and save frame
    annotated = draw_tracks(frame.copy(), current_keypoints, current_ids)
    cv2.imwrite(os.path.join(OUTPUT_DIR, f"frame_{frame_idx:04d}.png"), annotated)
    frame_idx += 1

cap.release()

# Generate output video
output_video_path = os.path.join("./Tracking/Output", "skeleton_tracked_video.mp4")
image_files = sorted([f for f in os.listdir(OUTPUT_DIR) if f.endswith(".png")])

if image_files:
    first_image = cv2.imread(os.path.join(OUTPUT_DIR, image_files[0]))
    height, width, _ = first_image.shape
    out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (width, height))

    for filename in image_files:
        frame = cv2.imread(os.path.join(OUTPUT_DIR, filename))
        out.write(frame)

    out.release()
    print(f"Tracking complete. Frames saved in: {OUTPUT_DIR}")
    print(f"Video saved as: {output_video_path}")
else:
    print("No frames were saved, so no video was generated.")
