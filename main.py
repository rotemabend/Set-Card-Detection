import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from find_cards import *
from find_properties_cnn import *
from show_properties import *
from show_set import *

#16
if __name__ == '__main__':
    image_path = os.path.join(BASE_DIR, "data/test/1.jpg")
    # Read the image
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)

    scale_percent = 50
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    cv2.imshow("Image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    contours = find_cards(img)
    cards_strings_map, cards_properties_map = find_properties()
    show_properties(img, contours, cards_strings_map)
    show_set(img, cards_properties_map, contours)


