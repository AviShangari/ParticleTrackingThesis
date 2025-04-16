import cv2
import numpy as np
import matplotlib.pyplot as plt

for i in range(3):
    # Load the image in grayscale
    img = "./curved-line-pics/Input/curved_line" + str(i+1) + ".jpg"
    image = cv2.imread(img, cv2.IMREAD_GRAYSCALE)

    # Apply Canny edge detection
    edges = cv2.Canny(image, 20, 55)

    # Find contours
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Convert grayscale image to color (so we can draw colored contours)
    contour_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    # Draw contours on the image
    cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 2)  # Green color, thickness=2

    # Save the result
    output = "./EdgeDetection/LineOutputs/contour_curved_line" + str(i+1) + ".jpg"
    cv2.imwrite(output, contour_image)