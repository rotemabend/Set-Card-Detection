import cv2
import numpy as np
import os
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from constants import *

def augment_data(X, y):
    # Create an ImageDataGenerator for data augmentation
    datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    # Fit the generator on the data
    datagen.fit(X)

    return datagen.flow(X, y, batch_size=32)

def train_model(X,y):
    X = np.array(X)
    y = np.array(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create an augmented data generator
    train_generator = augment_data(X_train, y_train)

    # Build the CNN model
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3)),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

        tf.keras.layers.Conv2D(64, (5, 5), activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

        tf.keras.layers.Conv2D(128, (5, 5), activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(len(set(y)), activation='softmax')  # Output layer for classification
    ])

    # Compile the model
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # Learning Rate Adjustment
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
    # Early Stopping
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    # Train the model using the augmented data generator
    model.fit(train_generator, epochs=600, validation_data=(X_test, y_test),
              callbacks=[reduce_lr, early_stopping])

    # Evaluate the model
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"Test Accuracy: {accuracy:.2f}")
    return model



X = []
y = []
for i in range(3):
    for j in range(3):
        for k in range(3):
            for l in range(3):
                path = os.path.join(BASE_DIR, "data/model_training_data/"+NUMBER[i]+"/"+COLOUR[j]+"/"+SHAPE[k]+"/"+CONTENT[l])
                for image_name in os.listdir(path):
                    image_path = os.path.join(path, image_name)
                    img =cv2.imread(image_path)
                    # Resize image
                    img = cv2.resize(img, IMAGE_SIZE)

                    # Normalize image data
                    img = img - img.mean()
                    img = img / 255.0  # Normalize to [0, 1]
                    X.append(img)
                    y.append(ENCODER_TWO_PROP[(i,k)])

model_number_shape = train_model(X,y_number_shape)
model_number_shape.save("cnn_model_number_shape.keras")
