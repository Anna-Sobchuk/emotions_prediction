import os
import cv2
import numpy as np
from skimage.feature import hog
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

# ---------------- CONFIG ----------------
TRAIN_DIR = "train"
MODEL_PATH = "models/hog_svm.joblib"

# ---------------- LOAD DATA ----------------
X_images, y_labels = [], []
for label in os.listdir(TRAIN_DIR):
    label_path = os.path.join(TRAIN_DIR, label)
    if not os.path.isdir(label_path):
        continue
    for file in os.listdir(label_path):
        if file.lower().endswith((".png",".jpg",".jpeg")):
            img = cv2.imread(os.path.join(label_path,file), cv2.IMREAD_GRAYSCALE)
            X_images.append(img)
            y_labels.append(label)

X_images = np.array(X_images)
y_labels = np.array(y_labels)
print("Loaded", len(X_images), "images, labels:", np.unique(y_labels))

# ---------------- FEATURE EXTRACTION ----------------
def hog_features(img):
    return hog(img, orientations=9, pixels_per_cell=(8,8), cells_per_block=(2,2), block_norm='L2-Hys', feature_vector=True)

X = np.array([hog_features(img) for img in X_images])

# ---------------- TRAIN ----------------
X_train, X_test, y_train, y_test = train_test_split(X, y_labels, test_size=0.2, random_state=42)
svc = LinearSVC(max_iter=10000)
svc_cal = CalibratedClassifierCV(svc, cv=5)
svc_cal.fit(X_train, y_train)

y_pred = svc_cal.predict(X_test)
print("HOG+SVM test accuracy:", accuracy_score(y_test, y_pred))

# ---------------- SAVE MODEL ----------------
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
joblib.dump(svc_cal, MODEL_PATH)
print(f"Model saved to {MODEL_PATH}")
