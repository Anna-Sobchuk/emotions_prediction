import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ---------------- CONFIG ----------------
TRAIN_DIR = "train"
MODEL_PATH = "models/mini_xception.h5"
IMG_SIZE = (48,48)
NUM_CLASSES = len(os.listdir(TRAIN_DIR))

# ---------------- LOAD DATA ----------------
X_images, y_labels = [], []
label_names = []

for idx,label in enumerate(os.listdir(TRAIN_DIR)):
    label_names.append(label)
    label_path = os.path.join(TRAIN_DIR,label)
    if not os.path.isdir(label_path):
        continue
    for file in os.listdir(label_path):
        if file.lower().endswith((".png",".jpg",".jpeg")):
            img = cv2.imread(os.path.join(label_path,file), cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, IMG_SIZE)
            X_images.append(img)
            y_labels.append(idx)

X = np.array(X_images, dtype='float32') / 255.0
X = X.reshape((-1, IMG_SIZE[0], IMG_SIZE[1], 1))
y = tf.keras.utils.to_categorical(y_labels, NUM_CLASSES)

# ---------------- SPLIT ----------------
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# ---------------- MODEL ----------------
def mini_xception(input_shape=(48,48,1), num_classes=NUM_CLASSES):
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv2D(8,(3,3),padding='same',activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(8,(3,3),padding='same',activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2,2))(x)
    
    x = layers.Conv2D(16,(3,3),padding='same',activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(16,(3,3),padding='same',activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2,2))(x)
    
    x = layers.Conv2D(32,(3,3),padding='same',activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling2D()(x)
    
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes,activation='softmax')(x)
    
    model = models.Model(inputs, outputs)
    return model

model = mini_xception()
model.compile(optimizer=optimizers.Adam(1e-3),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# ---------------- AUGMENTATION ----------------
datagen = ImageDataGenerator(rotation_range=10,
                             width_shift_range=0.1,
                             height_shift_range=0.1,
                             horizontal_flip=True)
datagen.fit(X_train)

# ---------------- TRAIN ----------------
model.fit(datagen.flow(X_train, y_train, batch_size=32),
          validation_data=(X_val, y_val),
          epochs=50)

# ---------------- SAVE ----------------
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
model.save(MODEL_PATH)
print(f"mini-Xception model saved to {MODEL_PATH}")
print("Class labels:", label_names)
