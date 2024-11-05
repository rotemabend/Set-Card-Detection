import cv2
import numpy as np
import os

import find_properties_cnn
from constants import *
def colour_detect(image):
    # Convert the image to the HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define color ranges (adjust these ranges as needed)
    red_lower = np.array([0, 120, 70])
    red_upper = np.array([10, 255, 255])
    green_lower = np.array([36, 100, 100])
    green_upper = np.array([86, 255, 255])
    purple_lower = np.array([125, 50, 50])
    purple_upper = np.array([155, 255, 255])

    # Create masks for each color
    red_mask = cv2.inRange(hsv_image, red_lower, red_upper)
    green_mask = cv2.inRange(hsv_image, green_lower, green_upper)
    purple_mask = cv2.inRange(hsv_image, purple_lower, purple_upper)

    # Calculate the number of non-zero pixels in each mask
    red_count = cv2.countNonZero(red_mask)
    green_count = cv2.countNonZero(green_mask)
    purple_count = cv2.countNonZero(purple_mask)

    # Determine the color with the largest count
    color_detected = ""
    if red_count > green_count and red_count > purple_count:
        color_detected = 2 #Red
    elif green_count > red_count and green_count > purple_count:
        color_detected = 1 #Green
    else:
        color_detected = 0 #Blue

    return color_detected

