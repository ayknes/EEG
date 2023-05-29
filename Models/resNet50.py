from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from PIL import Image
import numpy as np
import os
import sklearn as sk
from sklearn.model_selection import train_test_split

# Load the base pre-trained model
base_model = ResNet50(weights='imagenet', include_top=False)

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

# Split data into training and validation sets
train_images, val_images, train_labels, val_labels = train_test_split(images, labels, test_size=0.2, random_state=42)

# Train the model
history = model.fit(train_images, train_labels, validation_data=(val_images, val_labels), epochs=10)

# Save the history data for later analysis
with open('history_resNet50.csv', 'w') as f:
    f.write("epoch,loss,accuracy,val_loss,val_accuracy\n")
    for epoch in range(len(history.history['loss'])):
        f.write(f"{epoch},{history.history['loss'][epoch]},{history.history['accuracy'][epoch]},{history.history['val_loss'][epoch]},{history.history['val_accuracy'][epoch]}\n")
