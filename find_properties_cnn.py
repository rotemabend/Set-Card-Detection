from tensorflow.keras.models import load_model
import os
import cv2
import numpy as np
import tensorflow as tf
from collections import deque
from constants import *
from colour_detection import *

def white_edges(img):
    threshold = find_white_thresh(img)
    img = dfs_white(img,3,3, threshold)
    img = dfs_white(img,img.shape[0]-4,3, threshold)
    img = dfs_white(img,3, img.shape[1]-4, threshold)
    img = dfs_white(img,img.shape[0]-2, img.shape[1]-2, threshold)
    return img

def find_white_thresh(img):
    # Define the white color
    white = np.array([255, 255, 255])

    # Reshape the image to a 2D array of pixels
    pixels = img.reshape(-1, 3)

    # Calculate the squared Euclidean distance from white
    distances = np.linalg.norm(pixels - white, axis=1)

    # Get the number of pixels
    num_pixels = len(pixels)

    # Determine the number of closest pixels (20% of total)
    num_closest = int(num_pixels * 0.5)

    # Get indices of the closest pixels
    closest_indices = np.argsort(distances)[:num_closest]

    # Select the closest pixels
    closest_pixels = pixels[closest_indices]

    # Calculate the average color of the closest pixels
    average_color = closest_pixels.mean(axis=0)

    return average_color.astype(int)

def dfs_white(img,i,j, threshold):
    visited = set()
    stack = deque()
    stack.append((i, j))
    # threshold = [170,170,170]
    while (stack):
        (i,j) = stack.pop()
        visited.add((i,j))
        img[i][j] = threshold
        if (img[i+1][j][0]<threshold[0] and img[i+1][j][1]<threshold[1] and img[i+1][j][2]<threshold[2]) and (i+1,j) not in visited and i<img.shape[0]-2 and not (i>10 and i<img.shape[0]-10 and j>10 and j<img.shape[1]-10):
            stack.append((i+1,j))
        if (img[i][j+1][0]<threshold[0] and img[i][j+1][1]<threshold[1] and img[i][j+1][2]<threshold[2]) and (i,j+1) not in visited and j<img.shape[1]-2 and not (i>10 and i<img.shape[0]-10 and j>10 and j<img.shape[1]-10):
            stack.append((i,j+1))
        if (img[i-1][j][0]<threshold[0] and img[i-1][j][1]<threshold[1] and img[i-1][j][2]<threshold[2]) and (i-1,j) not in visited and i>0 and not (i>10 and i<img.shape[0]-10 and j>10 and j<img.shape[1]-10):
            stack.append((i-1,j))
        if (img[i][j-1][0]<threshold[0] and img[i][j-1][1]<threshold[1] and img[i][j-1][2]<threshold[2]) and (i,j-1) not in visited and j>0 and not (i>10 and i<img.shape[0]-10 and j>10 and j<img.shape[1]-10):
            stack.append((i,j-1))
    return img

def fix_order(lst):
    # moving cards 10,11 from indices 2,3 to the end of the list
    if len(lst) == 12:
        card10 = lst[2]
        card11 = lst[3]
        for i in range(4,12):
            lst[i-2] = lst[i]
        lst[10] = card10
        lst[11] = card11
        return lst
    # moving cards 10 from indice 2 to the end of the list
    if len(lst) == 11:
        card10 = lst[2]
        for i in range(3,11):
            lst[i-1] = lst[i]
        lst[10] = card10
        return lst
    return lst




def get_predictions(model, show_white_edges = False):
    path = os.path.join(BASE_DIR, "data/extracted_cards")
    cards = []
    for image_name in os.listdir(path):
        image_path = os.path.join(path, image_name)
        img = cv2.imread(image_path)
        # make the edges white
        img = white_edges(img)
        if show_white_edges:
            cv2.imshow("after white", img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        # Resize image
        img = cv2.resize(img, IMAGE_SIZE)
        # Normalize image data
        img = img - img.mean()
        img = img / 255.0  # Normalize to [0, 1]
        cards.append(img)

    # bringing cards 10,11 to the end of the list
    cards = fix_order(cards)

    cards = np.array(cards)
    predictions = model.predict(cards)

    # Get the predicted class for each image
    predicted_classes = np.argmax(predictions, axis=1)
    return predicted_classes


def predict_colour():
    path = os.path.join(BASE_DIR, "data/extracted_cards")
    cards = []
    for image_name in os.listdir(path):
        image_path = os.path.join(path, image_name)
        img = cv2.imread(image_path)
        cards.append(img)
    cards = fix_order(cards)
    colours = []
    for img in cards:
        colour = colour_detect(img)
        colours.append(colour)
    return colours


def find_properties(show_white_edges = False):
    # Load the model
    cnn_model_number_shape = load_model(os.path.join(BASE_DIR, "best_models/cnn_model_number_shape.keras"))
    content_model = load_model(os.path.join(BASE_DIR, "best_models/cnn_model_content.keras"))
    predictions_number_shape = get_predictions(cnn_model_number_shape, show_white_edges)
    predictions_colour = predict_colour()
    predictions_content = get_predictions(content_model)
    cards_strings_map = {}
    cards_properties_map = {}
    for i in range(len(predictions_number_shape)):
        card_number, card_shape = DECODER_TWO_PROP[predictions_number_shape[i]]
        card_colour = predictions_colour[i]
        card_content = predictions_content[i]
        cards_strings_map[i] = NUMBER[card_number]+" "+COLOUR[card_colour]+" "+SHAPE[card_shape]+" "+CONTENT[card_content]
        cards_properties_map[(card_number,card_colour,card_shape, card_content)] = i
        # print(NUMBER[card_number]+" "+COLOUR[card_colour]+" "+SHAPE[card_shape]+" "+CONTENT[card_content])

    return cards_strings_map, cards_properties_map