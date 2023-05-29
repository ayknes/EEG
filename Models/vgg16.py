from keras.applications.vgg16 import VGG16, preprocess_input
from keras.optimizers import RMSprop
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
import pandas as pd
import keras as keras
from PIL import Image
import numpy as np
import os
import sklearn as sk
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

# Load the base pre-trained model
base_model = VGG16(weights='imagenet', include_top=False)

# Add a global spatial average pooling layer
x = base_model.output
x = GlobalAveragePooling2D()(x)

# Add a fully-connected layer
x = Dense(1024, activation='relu')(x)

# Add a logistic layer -- we have 2 classes (right and left hand movements)
predictions = Dense(2, activation='softmax')(x)

# This is the model we will train
model = Model(inputs=base_model.input, outputs=predictions)

# First: train only the top layers (which were randomly initialized)
for layer in base_model.layers:
    layer.trainable = False

# Compile the model (should be done *after* setting layers to non-trainable)
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

# Now, we can load and preprocess our images
right_image_dir = '/home/ayknes/EEG/Data/drive-download-20230528T212437Z-001/droite_dom'  # replace with the path to your right hand images
left_image_dir = '/home/ayknes/EEG/Data/drive-download-20230528T212437Z-001/gauche_dom'  # replace with the path to your left hand images
right_image_files = os.listdir(right_image_dir)
left_image_files = os.listdir(left_image_dir)

# Initialize an empty array for our images and labels
images = []
labels = []

# Load and preprocess each image
# Load and preprocess each image
for image_file in right_image_files:
    img = Image.open(os.path.join(right_image_dir, image_file)).convert('RGB')  # convert to RGB
    img = img.resize((224, 224))
    img_array = np.array(img)
    img_array = preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)
    images.append(img_array)
    labels.append([1, 0])  # right hand movement is class [1, 0]

for image_file in left_image_files:
    img = Image.open(os.path.join(left_image_dir, image_file)).convert('RGB')  # convert to RGB
    img = img.resize((224, 224))
    img_array = np.array(img)
    img_array = preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)
    images.append(img_array)
    labels.append([0, 1])  # left hand movement is class [0, 1]
# Convert our list of images and labels to a numpy array
images = np.concatenate(images)
labels = np.array(labels)

# Shuffle your data
images, labels = shuffle(images, labels, random_state=42)

# Split data into training and validation sets
train_images, val_images, train_labels, val_labels = train_test_split(images, labels, test_size=0.2, random_state=42)

# Train the model
model.fit(train_images, train_labels, validation_data=(val_images, val_labels), epochs=10)

# Unfreeze some layers in the convolutional base
for layer in base_model.layers[:15]:
    layer.trainable = False
for layer in base_model.layers[15:]:
    layer.trainable = True

# Re-compile the model with a lower learning rate
model.compile(optimizer=RMSprop(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(train_images, train_labels, validation_data=(val_images, val_labels), epochs=12)

# Access the history object's data
print(history.history)

# Convert the history.history dict to a pandas DataFrame:     
hist_df = pd.DataFrame(history.history) 

# Save to csv: 
hist_csv_file = 'history_vgg16.csv'
with open(hist_csv_file, mode='w') as f:
    hist_df.to_csv(f)