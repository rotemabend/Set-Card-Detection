import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def find_set(cards_properties_map):
    def generate_feature3(feature1, feature2):
        feature_lst = [i for i in [0, 1, 2] if i != feature1 and i != feature2]
        if len(feature_lst) == 2:
            return feature1
        else:
            return feature_lst[0]

    def generate_card3(card1, card2):
        number = generate_feature3(card1[0], card2[0])
        colour = generate_feature3(card1[1], card2[1])
        shape = generate_feature3(card1[2], card2[2])
        content = generate_feature3(card1[3], card2[3])

        return (number, colour, shape, content)


    for card1 in cards_properties_map:
        for card2 in cards_properties_map:
            if card1 == card2:
                continue
            card3 = generate_card3(card1,card2)
            if card3 in cards_properties_map:
                return [cards_properties_map[card1], cards_properties_map[card2],cards_properties_map[card3]]
    # if there is no set, return empty list
    return []

def show_set(img,cards_properties_map, contours):
    set_lst = find_set(cards_properties_map)
    if set_lst == []: #if there is no set print "no set was found" on the image

        text = "no set was found"
        font = cv2.FONT_HERSHEY_SIMPLEX  # Choose a font
        font_scale = 1  # Font size (adjust as needed)
        color = (0, 255, 255)  # Green text
        thickness = 2  # Thickness of the text

        # Get image dimensions (height, width, channels)
        image_height, image_width = img.shape[:2]

        # Calculate the size of the text to center it
        (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)

        # Calculate the X coordinate for the text's top-center position
        text_x = (image_width - text_width) // 2
        text_y = 30  # Distance from the top (you can adjust this value as needed)

        # Put the text in the top-center of the image
        cv2.putText(img, text, (text_x, text_y), font, font_scale, color, thickness)

    else:
        i=0
        for contour in contours:
           if i in set_lst:
                x, y, w, h = cv2.boundingRect(contour)  # Get bounding box for the contour
                # Draw rectangle on the image for visualization (remove if unnecessary)
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
           i+=1
    # Show the result
    cv2.imshow("Detected Set", img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return