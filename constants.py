import os

def label_colour_content():
    labelencoder = {}
    labeldecoder = {}
    label = 0
    for i in range(3):
        for j in range(3):
            labelencoder[(i,j)] = label
            labeldecoder[label] = (i,j)
            label+=1
    return labelencoder, labeldecoder

def label_encoder_decoder():
    labelencoder = {}
    labeldecoder = {}
    label = 0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                for l in range(3):
                    labelencoder[(i, j,k,l)] = label
                    labeldecoder[label] = (i, j,k,l)
                    label += 1
    return labelencoder, labeldecoder

def label_encoder_decoder_two_prop():
    labelencoder = {}
    labeldecoder = {}
    label = 0
    for i in range(3):
        for j in range(3):
            labelencoder[(i, j)] = label
            labeldecoder[label] = (i, j)
            label += 1
    return labelencoder, labeldecoder

ENCODER, DECODER = label_encoder_decoder()
ENCODER_TWO_PROP, DECODER_TWO_PROP = label_encoder_decoder_two_prop()
LABELENCODER, LABELDECODER = label_colour_content()
IMAGE_SIZE = (105, 65)
NUMBER = ["one", "two", "three"]
COLOUR = ["blue", "green", "red"]
SHAPE = ["diamond", "oval", "squiggle"]
CONTENT = ["empty", "full", "partial"]
BASE_DIR = os.path.expanduser("~/PycharmProjects/SetDetection")

