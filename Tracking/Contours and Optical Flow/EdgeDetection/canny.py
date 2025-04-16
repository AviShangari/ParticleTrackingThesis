import cv2
import numpy as np
import matplotlib.pyplot as plt

for i in range(9):
    # Load the image in grayscale
    img = "./curved-line-pics/Input/curved_line" + str(i+1) + ".jpg"
    image = cv2.imread(img, cv2.IMREAD_GRAYSCALE)

    # Apply Gaussian Blur to reduce noise (optional but recommended)
    blurred = cv2.GaussianBlur(image, (5, 5), sigmaX=0)

    # Apply Canny Edge Detection
    edges = cv2.Canny(blurred, threshold1=20, threshold2=55)

    # Save the resulting image
    output = "./EdgeDetection/LineOutputs/curved_line" + str(i+1) + ".jpg"
    cv2.imwrite(output, edges)