import cv2
import numpy as np
import matplotlib.pyplot as plt

# === Your preprocessing function ===
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

# === Your extract_worm_masks function ===
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

# === Load one frame and apply steps ===
frame = cv2.imread('./Tracking/vid1frame.png')  # Replace with frame path
binary = preprocess_frame(frame)
masks = extract_worm_masks(binary)

# Combine all masks into one image for visualization
if masks:
    combined_mask = np.zeros_like(binary)
    for mask in masks:
        combined_mask = cv2.bitwise_or(combined_mask, mask * 255)
else:
    combined_mask = np.zeros_like(binary)

# === Display both side-by-side ===
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(binary, cmap='gray')
plt.title("After Preprocessing (Thresholded)")
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(combined_mask, cmap='gray')
plt.title("After Mask Extraction")
plt.axis('off')

plt.tight_layout()
plt.show()
