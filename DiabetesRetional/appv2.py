# import the necessary packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import cv2
from sklearn.model_selection import train_test_split

# load dataset
path ='/kaggle/input/diabetic-retinopathy-dataset/'
categories = os.listdir(path)

# teserflow and keras
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from sklearn.metrics import classification_report

# prepare data
data_dir = '/kaggle/input/diabetic-retinopathy-dataset/'
IMG_WIDTH, IMG_HEIGHT=64,64
categories = os.listdir(data_dir)
num_categories = len(categories)
X = []
y = []
for category in categories:
    category_dir = os.path.join(data_dir, category)
    for i, img_name in enumerate(os.listdir(category_dir)):
        img_path = os.path.join(category_dir, img_name)
        img = cv2.imread(img_path)
        img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
        X.append(img)
        y.append(categories.index(category))

X = np.array(X).astype('float32') / 255.0
y = np.array(y)

# split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,stratify = y)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

# build model

# Add the first convolutional layer
input_shape = (64,64,3)
num_classes = 5
input_shape = (64, 64, 3)
model = Sequential()

model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(256, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())

# Add the dense layers
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))

# Add the output layer
num_classes = len(set(y_train))
model.add(Dense(num_classes, activation='softmax'))

# Print the model summary
model.summary()

# compile model
from keras.utils import to_categorical
y_train_cat = to_categorical(y_train, num_classes)
y_test_cat = to_categorical(y_test, num_classes)

# compile model
import tensorflow as tf

# Define the optimizer
learning_rate = 0.001  
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
# Compile the model with the optimizer
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# train model
history = model.fit(X_train, y_train_cat, epochs=30, batch_size=32, validation_data=(X_test, y_test_cat))

# evaluate model
y_pred = model.predict(X_test)
y_pred_labels = np.argmax(y_pred, axis=1)
print(classification_report(y_test, y_pred_labels))

# save model
save_path = '/kaggle/working/'
model.save(save_path + 'v54')

# load model and predict
from tensorflow.keras.models import load_model
input_image_path = '/kaggle/input/diabetic-retinopathy-dataset/Healthy/Healthy_100.png'
input_image = cv2.imread(input_image_path)
input_image = cv2.resize(input_image, (64, 64))
input_image = np.expand_dims(input_image, axis=0)
input_image = input_image.astype('float32') / 255.0

saved_model_path = '/kaggle/working/v54'
model = load_model(saved_model_path)

prediction = model.predict(input_image)
predicted_class_index = np.argmax(prediction)
predicted_class = categories[predicted_class_index]

print("Predicted class:", predicted_class)

# load model and predict
from tensorflow.keras.models import load_model
input_image_path = '/kaggle/input/diabetic-retinopathy-dataset/Moderate DR/Moderate_DR_101.png'
input_image = cv2.imread(input_image_path)
input_image = cv2.resize(input_image, (64, 64))
input_image = np.expand_dims(input_image, axis=0)
input_image = input_image.astype('float32') / 255.0

saved_model_path = '/kaggle/working/v54'
model = load_model(saved_model_path)

prediction = model.predict(input_image)
predicted_class_index = np.argmax(prediction)
predicted_class = categories[predicted_class_index]

print("Predicted class:", predicted_class)

# load model and predict
from tensorflow.keras.models import load_model
input_image_path = '/kaggle/input/diabetic-retinopathy-dataset/Mild DR/Mild_DR_101.png'
input_image = cv2.imread(input_image_path)
input_image = cv2.resize(input_image, (64, 64))
input_image = np.expand_dims(input_image, axis=0)
input_image = input_image.astype('float32') / 255.0

saved_model_path = '/kaggle/working/v54'
model = load_model(saved_model_path)

prediction = model.predict(input_image)
predicted_class_index = np.argmax(prediction)
predicted_class = categories[predicted_class_index]

print("Predicted class:", predicted_class)

# load model and predict
from tensorflow.keras.models import load_model
input_image_path = '/kaggle/input/diabetic-retinopathy-dataset/Severe DR/Severe DR_100.png'
input_image = cv2.imread(input_image_path)
input_image = cv2.resize(input_image, (64, 64))
input_image = np.expand_dims(input_image, axis=0)
input_image = input_image.astype('float32') / 255.0

saved_model_path = '/kaggle/working/v54'
model = load_model(saved_model_path)

prediction = model.predict(input_image)
predicted_class_index = np.argmax(prediction)
predicted_class = categories[predicted_class_index]

print("Predicted class:", predicted_class)