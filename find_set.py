from tensorflow.keras.models import load_model
import os
import cv2
import numpy as np
import tensorflow as tf
IMAGE_SIZE = (105, 65)
from collections import deque
from constants import *

def white_edges(img):
    img = dfs_white(img,3,3)
    img = dfs_white(img,img.shape[0]-4,3)
    img = dfs_white(img,3, img.shape[1]-4)
    img = dfs_white(img,img.shape[0]-2, img.shape[1]-2)
    return img
def dfs_white(img,i,j):
    threshold = 150
    visited = set()
    stack = deque()
    stack.append((i, j))
    while (stack):
        (i,j) = stack.pop()
        visited.add((i,j))
        img[i][j] = [255, 255, 255]
        if (img[i+1][j][0]<threshold and img[i+1][j][1]<threshold and img[i+1][j][2]<threshold) and (i+1,j) not in visited and i<img.shape[0]-2 and not (i>17 and i<img.shape[0]-17 and j>17 and j<img.shape[1]-17):
            stack.append((i+1,j))
        if (img[i][j+1][0]<threshold and img[i][j+1][1]<threshold and img[i][j+1][2]<threshold) and (i,j+1) not in visited and j<img.shape[1]-2 and not (i>17 and i<img.shape[0]-17 and j>17 and j<img.shape[1]-17):
            stack.append((i,j+1))
        if (img[i-1][j][0]<threshold and img[i-1][j][1]<threshold and img[i-1][j][2]<threshold) and (i-1,j) not in visited and i>0 and not (i>17 and i<img.shape[0]-17 and j>17 and j<img.shape[1]-17):
            stack.append((i-1,j))
        if (img[i][j-1][0]<threshold and img[i][j-1][1]<threshold and img[i][j-1][2]<threshold) and (i,j-1) not in visited and j>0 and not (i>17000 and i<img.shape[0]-17 and j>17 and j<img.shape[1]-17):
            stack.append((i,j-1))
    return img



def build_encoders():
    label = 0
    propertiestolabel, labeltoproperties = {}, {}
    for i in range(3):
        for j in range(3):
            for k in range(3):
                for l in range(3):
                    propertiestolabel[(i, j, k, l)] = label
                    labeltoproperties[label] = (i, j, k, l)
                    label += 1
    return labeltoproperties, propertiestolabel

def get_predictions(model):
    path = "/Users/rotem/PycharmProjects/SetDetection/data/extracted_cards"
    cards = []
    for image_name in os.listdir(path):
        print(image_name)
        image_path = os.path.join(path, image_name)
        img = cv2.imread(image_path)
        # white edges
        img = white_edges(img)
        # cv2.imshow("after white", img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()


        # Set alpha and beta for contrast and brightness
        alpha = 2.0  # Contrast control (1.0-3.0)
        beta = 0  # Brightness control (0-100)
        # Adjust the contrast
        img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
        cv2.imshow("high contrast", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # Resize image
        img = cv2.resize(img, IMAGE_SIZE)
        # Normalize image data
        img = img - img.mean()
        img = img / 255.0  # Normalize to [0, 1]
        cards.append(img)


    cards = np.array(cards)
    predictions = model.predict(cards)

    # Get the predicted class for each image
    predicted_classes = np.argmax(predictions, axis=1)
    return predicted_classes

number = ["one", "two", "three"]
colour = ["blue", "green", "red"]
shape = ["diamond", "oval", "squiggle"]
content = ["empty", "full", "partial"]
# Load the model
# model = load_model("/Users/rotem/PycharmProjects/SetDetection/card_classifier_model.keras")
number_model = load_model("/Users/rotem/PycharmProjects/SetDetection/card_number_classifier_model.keras")
# colour_model = load_model("/Users/rotem/PycharmProjects/SetDetection/card_colour_classifier_model.keras")
shape_model = load_model("/Users/rotem/PycharmProjects/SetDetection/card_shape_classifier_model.keras")
# content_model = load_model("/Users/rotem/PycharmProjects/SetDetection/card_content_classifier_model.keras")
model_colour_content_1 = load_model("/Users/rotem/PycharmProjects/SetDetection/card_colour_content_1_classifier_model.keras")
model_colour_content_2 = load_model("/Users/rotem/PycharmProjects/SetDetection/card_colour_content_2_classifier_model.keras")
model_colour_content_3 = load_model("/Users/rotem/PycharmProjects/SetDetection/card_colour_content_3_classifier_model.keras")


number_predictions = get_predictions(number_model)
# colour_predictions = get_predictions(colour_model)
shape_predictions = get_predictions(shape_model)
# content_predictions = get_predictions(content_model)
colour_content_1_card_predictions = get_predictions(model_colour_content_1)
colour_content_2_card_predictions = get_predictions(model_colour_content_2)
colour_content_3_card_predictions = get_predictions(model_colour_content_3)


labeltoproperties, propertiestolabel = build_encoders()
for i in range(len(number_predictions)):
    card_number = number_predictions[i]
    # card_colour = colour[colour_predictions[i]]
    card_shape = shape_predictions[i]
    # card_content = content[content_predictions[i]]
    if card_number == 0:
        card_colour = LABELDECODER[colour_content_1_card_predictions[i]][0]
        card_content = LABELDECODER[colour_content_1_card_predictions[i]][1]
    elif card_number == 1:
        card_colour = LABELDECODER[colour_content_2_card_predictions[i]][0]
        card_content = LABELDECODER[colour_content_2_card_predictions[i]][1]
    else:
        card_colour = LABELDECODER[colour_content_3_card_predictions[i]][0]
        card_content = LABELDECODER[colour_content_3_card_predictions[i]][1]
    print(number[card_number]+" "+colour[card_colour]+" "+shape[card_shape]+" "+content[card_content])



