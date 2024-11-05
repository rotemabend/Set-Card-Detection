import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from constants import *
import glob

def clean_output_folder(folder_path):
    all_files = glob.glob(os.path.join(folder_path, '*'))
    # Loop through and delete each file
    for file in all_files:
        if os.path.isfile(file):  # Ensure it is a file, not a directory
            os.remove(file)

def find_avg_contour_area(contours):
    # Calculate the average width and height of contours
    total_width, total_height, count = 0, 0, 0
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        total_width += w
        total_height += h
        count += 1
    # Avoid division by zero
    if count > 0:
        average_width = total_width / count
        average_height = total_height / count
    else:
        average_width = 0
        average_height = 0

    # Set thresholds for filtering
    threshold = 0.4
    return average_width*average_height*threshold

def find_cards(img):
    #  clean the output folder
    output_folder = os.path.join(BASE_DIR, "data/extracted_cards")
    clean_output_folder(output_folder)

    edges = cv2.Canny(image= img, threshold1=100, threshold2=700)
    # Find contours in the edge image and save them
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    card_coordinates = []
    card_images = []
    card_count = 0

    area_thresh = find_avg_contour_area(contours)
    contours_res = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w*h>=area_thresh:
            card_image = img[y:y + h, x:x + w]
            card_coordinates.append((x, y, w, h))
            # Draw rectangle on the original image for visualization (if needed)
            # cv2.rectangle(sharpened_image, (x, y), (x + w, y + h), (255, 0, 0), 2)
            card_images.append(card_image)
            # print(os.path.join(output_folder, f"card_{card_count}.jpg"))
            cv2.imwrite(os.path.join(output_folder, f"card_{card_count}.jpg"), card_image)
            card_count += 1
            contours_res.append(contour)

    # Show the result
    # cv2.imshow("Detected Cards", sharpened_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return contours_res