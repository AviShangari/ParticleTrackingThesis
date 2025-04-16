import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize
from skimage.graph import route_through_array
from skimage.util import invert
from scipy.ndimage import convolve

# === CONFIG ===
VIDEO_FRAME_PATH = "./Tracking/vid1frame.png"  # Replace with frame image
AREA_THRESHOLD = 50
KEYPOINTS_PER_WORM = 15

# === Preprocessing Function ===
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

# === Mask Extraction Function ===
def extract_worm_masks(binary, area_threshold=50):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary)
    masks = []
    height = binary.shape[0]
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= area_threshold:
            y = stats[i, cv2.CC_STAT_TOP]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            if y + h >= height - 5:
                continue
            mask = (labels == i).astype(np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
            masks.append(mask)
    return masks

# === Skeleton + 15 Point Extraction Function ===
def get_skeleton_points(mask, num_points):
    mask_dilated = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1)
    skeleton = skeletonize(mask_dilated > 0)

    if np.count_nonzero(skeleton) < 2:
        return None, skeleton

    kernel = np.ones((3, 3), dtype=int)
    neighbor_count = convolve(skeleton.astype(int), kernel, mode='constant') * skeleton
    endpoints = np.column_stack(np.where(neighbor_count == 2))

    if len(endpoints) < 2:
        return None, skeleton

    start = tuple(endpoints[0])
    end = tuple(endpoints[-1])

    try:
        path, _ = route_through_array(invert(skeleton).astype(np.float32), start, end, fully_connected=True)
    except Exception:
        return None, skeleton

    path = np.array(path)
    if len(path) < 2:
        return None, skeleton

    indices = np.linspace(0, len(path) - 1, num=num_points, dtype=int)
    return path[indices], skeleton

# === Load and Process Frame ===
frame = cv2.imread(VIDEO_FRAME_PATH)
binary = preprocess_frame(frame)
masks = extract_worm_masks(binary)

# === Combine all masks ===
combined_mask = np.zeros_like(binary)
skeleton_overlay = np.zeros_like(binary)
all_points = []

for mask in masks:
    combined_mask = cv2.bitwise_or(combined_mask, mask * 255)
    points, skeleton = get_skeleton_points(mask, KEYPOINTS_PER_WORM)
    if skeleton is not None:
        skeleton_overlay = cv2.bitwise_or(skeleton_overlay, (skeleton * 255).astype(np.uint8))
    if points is not None:
        all_points.append(points)

# === Plotting ===
plt.figure(figsize=(10, 5))

# Left: Pre-skeleton mask
plt.subplot(1, 2, 1)
plt.imshow(combined_mask, cmap='gray')
plt.title("Worm Mask (Pre-Skeleton)")
plt.axis("off")

# Right: Skeleton with points
plt.subplot(1, 2, 2)
plt.imshow(skeleton_overlay, cmap='gray')
for worm_points in all_points:
    for pt in worm_points:
        plt.plot(pt[1], pt[0], 'ro', markersize=3)
plt.title("Skeleton + 15 Points Per Worm")
plt.axis("off")

plt.tight_layout()
plt.show()
