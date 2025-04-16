import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the video
video_path = "./Tracking/vid1.mov" 
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

frame_num = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_num += 1

    if frame_num % 10 == 0:
        # Original frame (for display)
        original_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Convert to grayscale and smooth it
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 2)

        # Canny edge detection
        edges = cv2.Canny(blurred, 30, 60)

        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Draw contours on grayscale
        contour_frame = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        cv2.drawContours(contour_frame, contours, -1, (0, 255, 0), 2)

        # Prepare all three images for matplotlib (convert to RGB)
        edges_rgb = cv2.cvtColor(cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR), cv2.COLOR_BGR2RGB)
        contour_rgb = cv2.cvtColor(contour_frame, cv2.COLOR_BGR2RGB)

        # Plot side-by-side
        fig, axs = plt.subplots(1, 3, figsize=(16, 6))
        axs[0].imshow(original_rgb)
        axs[0].set_title("Original Frame")
        axs[0].axis('off')

        axs[1].imshow(edges_rgb)
        axs[1].set_title("Canny Edges")
        axs[1].axis('off')

        axs[2].imshow(contour_rgb)
        axs[2].set_title("Contours")
        axs[2].axis('off')

        plt.tight_layout()
        plt.show()