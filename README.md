# Set Card Detection Project
This project focuses on the use of computer vision and deep learning to detect and classify cards from the **SET** game within a board image. The workflow includes identifying card regions using computer vision techniques and employing convolutional neural networks (CNNs) for accurate classification.
![1](https://github.com/user-attachments/assets/9d6b59c3-b3a0-48c1-b587-0ebe7dbd2d11)


## Card Detection
The card detection process utilizes **OpenCV** and follows these steps:

1. **Edge Detection**: Convert the image to an edge map to highlight card boundaries.
   ![edges image](https://github.com/user-attachments/assets/df5b37fc-be9a-4c67-9fa7-32555a37eb61)
2. **Contour Extraction**: Extract all contours from the image and filter them based on size, selecting those within a range of the average size * 0.6 to isolate the twelve cards.
 ![card_5](https://github.com/user-attachments/assets/f798cd89-fffc-47bb-b995-ef70c00c2789)
3. **Preprocessing**: Fill the edges of card images with white pixels to enhance compatibility with CNN input.
![after white](https://github.com/user-attachments/assets/125e3204-e26d-4404-9c44-30113942288b)


## Card Classification
For each detected card, the following steps were performed:

### 1. Color Detection
Using OpenCV, a color mask was applied to detect the predominant color (red, green, or purple) on each card. Pixel counts for each color were measured, and the most frequent color was selected.

### 2. Number, Shading, and Size Detection
Two separate CNNs were utilized:

1. **CNN 1**: Classifies the number of shapes and shape type.
2. **CNN 2**: Classifies the shading pattern (solid, striped, or outlined).

## CNN Training
The CNNs were trained using a dataset of card images sourced from Kaggle's SET Card dataset (https://www.kaggle.com/datasets/kwisatzhaderach/set-cards). This dataset provided diverse examples that were essential for accurate feature detection and classification.
![cnn model training](https://github.com/user-attachments/assets/58acd1b6-764b-47a7-b1de-b89eea967842)

## Results
![2](https://github.com/user-attachments/assets/69cfd599-8f93-466a-8d46-55fdb36e5359)
![3](https://github.com/user-attachments/assets/80fd3c8a-0fa0-4cac-8085-93551d91ef56)
![no set was found](https://github.com/user-attachments/assets/ca0a79a2-c388-42ee-95bf-05b504e52db7)

