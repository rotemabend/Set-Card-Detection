import cv2
import numpy as np
import matplotlib.pyplot as plt
import os


def show_properties(img, contours, cards_map):
    card_coordinates = []  # Initialize the list to store coordinates
    card_count = 0  # Initialize the card count

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)  # Get bounding box for the contour
        card_image = img[y:y + h, x:x + w]  # Crop the card image
        card_coordinates.append((x, y, w, h))  # Store the coordinates


        # Add text annotation at the top-left corner of each card
        text = f"{cards_map[card_count]}"
        font = cv2.FONT_HERSHEY_SIMPLEX  # Choose a font
        font_scale = 0.4  # Font size (adjust as needed)
        color = (155, 155, 155)  # Black text
        thickness = 1  # Thickness of the text

        # Place the text near the top-left corner of the card
        cv2.putText(img, text, (x, y - 10), font, font_scale, color, thickness)
        card_count+=1
    # Show the result
    cv2.imshow("Detected Cards", img)
    # for idx, card in enumerate(card_images):
    #     cv2.imshow(f"Card {idx}", card)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return