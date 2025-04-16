import cv2
import numpy as np
import matplotlib.pyplot as plt

def extract_key_points(image_path, min_points=2, max_points=5, min_contour_area=500):
    """
    Extracts key points from entities in an image and visualizes them.

    Parameters:
        image_path (str): Path to the input image.
        min_points (int): Minimum number of key points per entity.
        max_points (int): Maximum number of key points per entity.
        min_contour_area (int): Minimum area of contours to be considered (removes noise).
    
    Returns:
        list: A list of arrays, each containing key points for an entity.
    """
    # Load image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    if image is None:
        print("Error: Could not load image. Check the file path.")
        return []

    # Apply adaptive threshold for better object detection
    binary = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 11, 2)

    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    if not contours:
        print("Error: No contours found. Try adjusting the thresholding method.")
        return []

    key_points_list = []
    
    # Convert grayscale image to color for visualization
    image_color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    # Process each detected entity
    for contour in contours:
        if cv2.contourArea(contour) < min_contour_area:
            continue  # Ignore small noisy contours
        
        contour = contour.squeeze()
        if contour.ndim != 2:
            continue

        # Compute cumulative arc length for even spacing
        arc_lengths = np.zeros(len(contour))
        for i in range(1, len(contour)):
            arc_lengths[i] = arc_lengths[i - 1] + np.linalg.norm(contour[i] - contour[i - 1])
        
        # Determine the number of points based on contour length
        num_points = min(max(min_points, len(contour) // 30), max_points)
        
        # Select evenly spaced points along the contour
        selected_arc_lengths = np.linspace(0, arc_lengths[-1], num_points)
        key_points = []
        for arc_len in selected_arc_lengths:
            index = np.searchsorted(arc_lengths, arc_len)
            key_points.append(contour[index])
        key_points = np.array(key_points)
        key_points_list.append(key_points)

        # Draw key points
        for point in key_points:
            cv2.circle(image_color, tuple(point), 5, (0, 0, 255), -1)

    # Display the result
    plt.imshow(cv2.cvtColor(image_color, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

    return key_points_list

# Example usage
image_path = './curved-line-pics/Input/curved_line2.jpg' 
key_points = extract_key_points(image_path, min_points=9, max_points=10, min_contour_area=500)
print("Extracted Key Points:", key_points)
